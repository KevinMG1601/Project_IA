import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import cv2
from PIL import Image

# Reproducibilidad
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODEL = ROOT / 'model'
REPORTS = MODEL / 'reports'

# ---------------------- Utilidades ----------------------

def load_image(path: Path):
	img = cv2.imread(str(path))
	if img is None:
		raise RuntimeError(f'No se puede leer imagen: {path}')
	return img


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
	pages = []
	try:
		from pdf2image import convert_from_path
		pages = convert_from_path(str(pdf_path), dpi=dpi)
		return pages
	except Exception:
		try:
			import fitz  # PyMuPDF
			doc = fitz.open(str(pdf_path))
			for i, page in enumerate(doc):
				pix = page.get_pixmap(dpi=dpi)
				img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
				pages.append(img)
			return pages
		except Exception as e2:
			raise RuntimeError(f'No se pudo convertir PDF: {e2}')


def pil_to_cv(img: Image.Image):
	return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def normalize_money(x):
	if x is None:
		return None
	if isinstance(x, (int, float)):
		return float(x)
	s = str(x).strip().replace(' ', '')
	s = s.replace('.', '').replace(',', '.') if s.count(',') == 1 and s.count('.') > 1 else s
	s = re.sub(r'[^0-9\.-]', '', s)
	try:
		return float(s)
	except Exception:
		return None


def nit_check_colombia(nit: str) -> bool:
	if not nit:
		return False
	s = re.sub(r'[^0-9-]', '', nit)
	parts = s.split('-')
	if len(parts) != 2:
		return False
	num, dv = parts[0], parts[1]
	if not (num.isdigit() and dv.isdigit()):
		return False
	# Cálculo DV módulo 11 (DIAN)
	weights = [71,67,59,53,47,43,41,37,29,23,19,17,13,7,3]
	num_rev = list(map(int, num[::-1]))
	total = 0
	for i, d in enumerate(num_rev):
		w = weights[i] if i < len(weights) else 1
		total += d * w
	dv_calc = 11 - (total % 11)
	if dv_calc in (10, 11):
		dv_calc = 0
	return str(dv_calc) == dv


def format_date_iso(s: str):
	if not s:
		return None
	s = s.strip()
	s = s.replace('/', '-').replace('.', '-')
	m = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', s)
	return m.group(0) if m else None


# ---------------------- OCR ----------------------

def ocr_with_fallback(image_path: Path) -> Dict[str, Any]:
	"""Devuelve {'words':[{'text','bbox','conf'}], 'lines':[{'text'}], 'size':(w,h)}
	BBoxes normalizados a 0..1000 como LayoutLMv3.
	"""
	img = cv2.imread(str(image_path))
	if img is None:
		raise RuntimeError(f'No se puede abrir {image_path}')
	h, w = img.shape[:2]
	# Intentar PaddleOCR
	try:
		from paddleocr import PaddleOCR
		ocr = PaddleOCR(lang='es', use_angle_cls=True, show_log=False)
		res = ocr.ocr(str(image_path), cls=True)
		words, lines = [], []
		for block in res:
			for box, (text, conf) in block:
				xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
				x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
				bbox = [int(1000*x1/w), int(1000*y1/h), int(1000*x2/w), int(1000*y2/h)]
				words.append({'text': text, 'bbox': bbox, 'conf': float(conf)})
		line_text = ' '.join([w['text'] for w in words])
		lines.append({'text': line_text})
		return {'words': words, 'lines': lines, 'size': (w, h)}
	except Exception:
		pass
	# Fallback: Tesseract
	try:
		import pytesseract
		data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='spa')
		words, lines_map = [], {}
		for i in range(len(data['text'])):
			txt = data['text'][i]
			if not txt or str(txt).strip()=='' or txt=='-1':
				continue
			x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
			bbox = [int(1000*x/w), int(1000*y/h), int(1000*(x+bw)/w), int(1000*(y+bh)/h)]
			conf = data.get('conf',["0"]) [i]
			words.append({'text': txt, 'bbox': bbox, 'conf': float(conf) if str(conf).replace('.','',1).isdigit() else 0.0})
			line_num = data.get('line_num',[1])[i]
			lines_map.setdefault(line_num, []).append(txt)
		lines = [{'text': ' '.join(v)} for _, v in sorted(lines_map.items())]
		return {'words': words, 'lines': lines, 'size': (w, h)}
	except Exception as e2:
		raise RuntimeError(f'OCR falló: {e2}')


# ---------------------- Extracción por reglas ----------------------

def extract_fields_from_lines(lines: List[str]) -> Dict[str, Any]:
	text = '\n'.join(lines)
	# fecha
	fecha = None
	m = re.search(r'(\d{4}[-/.]\d{2}[-/.]\d{2})', text)
	if m:
		fecha = format_date_iso(m.group(1))
	# nit
	nit = None
	m = re.search(r'(\d{6,10}[-–]\d)', text)
	if m:
		nit = m.group(1).replace('–','-')
	# valores por anclas
	def find_value_after(keyword):
		for ln in lines:
			if keyword in ln.upper():
				m = re.search(r'([0-9\.,]+)', ln)
				if m:
					return normalize_money(m.group(1))
		return None
	total = find_value_after('TOTAL')
	subtotal = find_value_after('SUBTOTAL')
	iva_porcentaje = None
	m = re.search(r'IVA\s*(\d{1,2})\s*%|IVA\s*%\s*(\d{1,2})', text.upper())
	if m:
		iva_porcentaje = int([g for g in m.groups() if g][0])
	iva_valor = find_value_after('IVA')
	# razon_social (heurística)
	razon = None
	for ln in lines:
		up = ln.upper()
		if 'RAZON' in up or 'PROVEEDOR' in up or 'CLIENTE' in up:
			if len(ln.split()) >= 2:
				razon = ln
				break
	return {
		'fecha': fecha,
		'nit': nit,
		'razon_social': razon,
		'subtotal': subtotal,
		'iva_porcentaje': iva_porcentaje,
		'iva_valor': iva_valor,
		'total': total
	}


# ---------------------- Reconstrucciones y validaciones ----------------------

def reconstruct_iva(fields: Dict[str, Any]) -> Dict[str, Any]:
	subtotal = normalize_money(fields.get('subtotal'))
	total = normalize_money(fields.get('total'))
	iva_valor = normalize_money(fields.get('iva_valor'))
	iva_porcentaje = fields.get('iva_porcentaje')
	flag_recon = False
	fuente = 'explicita'
	if subtotal is not None and total is not None:
		if iva_valor is None and iva_porcentaje is not None:
			iva_valor = round(subtotal * (float(iva_porcentaje)/100.0), 2)
			flag_recon = True; fuente = 'pct'
		elif iva_porcentaje is None and iva_valor is not None and subtotal > 0:
			iva_porcentaje = int(round(100.0 * iva_valor / subtotal))
			flag_recon = True; fuente = 'valor'
		elif iva_valor is None and iva_porcentaje is None:
			iva_valor = round(max(0.0, total - subtotal), 2)
			iva_porcentaje = int(round(100.0 * iva_valor / subtotal)) if subtotal else None
			flag_recon = True; fuente = 'delta'
	fields['iva_valor'] = iva_valor
	fields['iva_porcentaje'] = iva_porcentaje
	return {'iva_reconstruido': bool(flag_recon), 'fuente_iva': fuente}


def total_check(fields: Dict[str, Any]) -> bool:
	subtotal = normalize_money(fields.get('subtotal'))
	iva_val = normalize_money(fields.get('iva_valor'))
	total = normalize_money(fields.get('total'))
	if None in [subtotal, iva_val, total]:
		return False
	return abs(subtotal + iva_val - total) < 1.01


# ---------------------- API principal ----------------------

def predict(input_paths: List[str]) -> Dict[str, Any]:
	start = time.time()
	docs = []
	warnings = []
	for ip in input_paths:
		p = Path(ip)
		pages_cv = []
		if p.suffix.lower() == '.pdf':
			try:
				pages = pdf_to_images(p, dpi=300)
				pages_cv = [pil_to_cv(pg) for pg in pages]
			except Exception as e:
				warnings.append(f'PDF no convertido: {p.name} -> {e}')
		else:
			pages_cv = [load_image(p)]
		# OCR y extracción por página
		all_lines = []
		for i, img in enumerate(pages_cv, start=1):
			page_path = p if len(pages_cv)==1 else Path(str(p).replace(p.suffix, f'_p{i:02d}.png'))
			cv2.imwrite(str(page_path), img) if not page_path.exists() and len(pages_cv)>1 else None
			ocr = ocr_with_fallback(page_path)
			all_lines.extend([ln.get('text','') for ln in ocr.get('lines',[]) if ln.get('text')])
		fields = extract_fields_from_lines(all_lines)
		recon_info = reconstruct_iva(fields)
		nit_ok = nit_check_colombia(fields.get('nit')) if fields.get('nit') else False
		status = 'ok'
		if not total_check(fields):
			status = 'warnings'
			d = {'total_check': False}
			warnings.append(f"TOTAL no cuadra para {p.name}")
		docs.append({
			'input': str(p),
			'campos': {
				'fecha': format_date_iso(fields.get('fecha')),
				'nit': fields.get('nit'),
				'razon_social': fields.get('razon_social'),
				'subtotal': normalize_money(fields.get('subtotal')),
				'iva_porcentaje': int(fields.get('iva_porcentaje')) if fields.get('iva_porcentaje') is not None else None,
				'iva_valor': normalize_money(fields.get('iva_valor')),
				'total': normalize_money(fields.get('total')),
			},
			'validaciones': {
				'nit_dv_ok': bool(nit_ok),
				'total_check': bool(total_check(fields)),
				'iva_reconstruido': bool(recon_info['iva_reconstruido']),
				'fuente_iva': recon_info['fuente_iva']
			},
			'status': status
		})
	elapsed = time.time() - start
	return {'results': docs, 'warnings': warnings, 'elapsed_s': elapsed}


# ---------------------- CLI ----------------------
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Inferencia unificada de facturas')
	parser.add_argument('--input', nargs='+', required=True, help='Rutas a PDF/JPG/PNG')
	parser.add_argument('--output', required=False, help='Ruta de salida JSON')
	args = parser.parse_args()

	res = predict(args.input)
	out_path = args.output
	if out_path:
		Path(out_path).parent.mkdir(parents=True, exist_ok=True)
		with open(out_path, 'w', encoding='utf-8') as f:
			json.dump(res, f, ensure_ascii=False, indent=2)
		print(f'Guardado: {out_path}')
	else:
		print(json.dumps(res, ensure_ascii=False))
