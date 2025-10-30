from pathlib import Path
import json
import cv2


def ocr_document_images(doc_id: str, page_paths: list, out_dir: Path) -> Path:
	"""Ejecuta OCR para una lista de páginas (imágenes) y guarda JSON compatible con LayoutLMv3.

	- Intenta PaddleOCR (si está instalado). Si falla, usa Tesseract (spa o eng).
	- BBoxes normalizados a 0..1000.
	- out_dir: carpeta de salida (e.g., data/ocr/)
	"""
	use_paddle = False
	ocr = None
	try:
		from paddleocr import PaddleOCR  # type: ignore
		ocr = PaddleOCR(lang='es', use_angle_cls=True, show_log=False)
		use_paddle = True
	except Exception:
		pass

	results = []
	for pth in page_paths:
		img = cv2.imread(str(pth))
		if img is None:
			continue
		h, w = img.shape[:2]
		if use_paddle and ocr is not None:
			res = ocr.ocr(str(pth), cls=True)
			words = []
			lines = []
			for block in res:
				for box, (text, conf) in block:
					xs = [pt[0] for pt in box]
					ys = [pt[1] for pt in box]
					x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
					bbox = [int(1000*x1/w), int(1000*y1/h), int(1000*x2/w), int(1000*y2/h)]
					words.append({'text': text, 'bbox': bbox, 'conf': float(conf)})
				line_text = ' '.join([w_['text'] for w_ in words])
				lines.append({'text': line_text})
			results.append({'page_path': str(pth), 'width': w, 'height': h, 'words': words, 'lines': lines})
		else:
			try:
				import pytesseract  # type: ignore
				# Preferir spa; fallback a eng
				try:
					data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='spa')
				except Exception:
					data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='eng')
				words = []
				lines_map = {}
				for i in range(len(data['text'])):
					txt = data['text'][i]
					if not txt or str(txt).strip()=='' or txt=='-1':
						continue
					x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
					bbox = [int(1000*x/w), int(1000*y/h), int(1000*(x+bw)/w), int(1000*(y+bh)/h)]
					conf = data.get('conf', ["0"]) [i]
					words.append({'text': txt, 'bbox': bbox, 'conf': float(conf) if str(conf).replace('.','',1).isdigit() else 0.0})
					line_num = data.get('line_num', [1])[i]
					lines_map.setdefault(line_num, []).append(txt)
				lines = [{'text': ' '.join(v)} for _, v in sorted(lines_map.items())]
				results.append({'page_path': str(pth), 'width': w, 'height': h, 'words': words, 'lines': lines})
			except Exception:
				# Saltar página si OCR falla
				continue

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f'{doc_id}.json'
	with open(out_path, 'w', encoding='utf-8') as f:
		json.dump({'doc_id': doc_id, 'pages': results}, f, ensure_ascii=False)
	return out_path


# 1) tools/ocr_utils.py
from pathlib import Path
import sys
ROOT = Path.cwd()
if (ROOT.name == 'notebooks') and (ROOT.parent / 'tools').exists():
    ROOT = ROOT.parent
sys.path.append(str(ROOT/'tools'))
from ocr_utils import ocr_document_images
print('OK: ocr_utils importado')

# 2) PaddleOCR disponible?
try:
    import paddleocr; print('PaddleOCR: OK')
except Exception as e:
    print('PaddleOCR: NO ->', e)

# 3) Tesseract y lenguajes
try:
    import pytesseract, subprocess
    print('Tesseract:', pytesseract.get_tesseract_version())
    print('Langs:', subprocess.check_output(['tesseract','--list-langs'], text=True))
except Exception as e:
    print('Tesseract check error:', e)