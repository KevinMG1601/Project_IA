# **PROYECTO DE GESTION CLOUD DE FACTURAS - KEVIN MUÑOZ🚛**

## Descripción
Este proyecto tiene como finalidad transformar digitalmente el proceso de gestión de facturas en empresas del sector transporte, utilizando técnicas de Inteligencia Artificial (IA) para digitalizar, extraer y registrar automáticamente los datos contables, eliminando tareas manuales repetitivas y errores humanos frecuentes.

## Objetivo general

Desarrollar un sistema inteligente capaz de recibir facturas y comprobantes de gasto (en papel, escaneo o foto), procesarlos automáticamente, extraer la información clave (número de factura, valor, fecha, NIT, entre otros) y alimentar, sin intervención manual, los registros en el sistema contable de la empresa.

- **Automatización:** Reduce drásticamente el tiempo y esfuerzo dedicado a la gestión manual de facturas.
- **Digitalización:** Facilita la transición de documentos físicos o imágenes a un registro contable digital.
- **Precisión:** Disminuye la probabilidad de errores humanos en la digitación.
- **Eficiencia:** Mejora la trazabilidad y el flujo de la información financiera dentro de la empresa.

## Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- Tesseract OCR instalado en el sistema
  - **Windows:** Descargar desde [GitHub Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) o usar `choco install tesseract`
  - **Linux:** `sudo apt-get install tesseract-ocr`
  - **macOS:** `brew install tesseract`

### Instalación de Dependencias

1. Instalar las dependencias de Python desde el archivo de requisitos:

```bash
pip install -r model/requirements.txt
```

### Iniciar el API

El API se puede iniciar de dos formas:

#### Opción 1: Usando el script de inicio (Recomendado)

```bash
python start_api.py
```

#### Opción 2: Ejecutando directamente el archivo api.py

```bash
python api.py
```

#### Configuración de Puerto y Host (Opcional)

Puedes configurar el puerto y host usando variables de entorno:

```bash
# Windows PowerShell
$env:PORT=8080
$env:HOST="127.0.0.1"
python start_api.py

# Linux/macOS
PORT=8080 HOST=127.0.0.1 python start_api.py
```

Por defecto, el API se ejecuta en:
- **Host:** `0.0.0.0` (todas las interfaces)
- **Puerto:** `8000`

### Acceso al API

Una vez iniciado, el API estará disponible en:

- **API Base:** `http://localhost:8000`
- **Documentación Interactiva (Swagger):** `http://localhost:8000/docs`
- **Documentación Alternativa (ReDoc):** `http://localhost:8000/redoc`
- **Health Check:** `http://localhost:8000/health`

## Endpoints Disponibles

### `GET /`
Información general sobre la API y endpoints disponibles.

### `GET /health`
Verifica el estado de salud del servicio. Retorna `{"status": "ok", "service": "invoice-extraction-api"}`

### `POST /extract`
Extrae datos de una factura desde una imagen o PDF.

**Parámetros:**
- `file`: Archivo de imagen (JPG, PNG) o PDF de la factura (multipart/form-data)

**Respuesta:**
```json
{
  "success": true,
  "campos": {
    "fecha": "...",
    "nit": "...",
    "razon_social": "...",
    "subtotal": "...",
    "iva": "...",
    "total": "..."
  },
  "validaciones": {
    "nit": true/false,
    "total": true/false,
    "iva": true/false
  },
  "status": "ok",
  "tiempo_procesamiento": 1.234,
  "warnings": []
}
```

**Ejemplo de uso con cURL:**
```bash
curl -X POST "http://localhost:8000/extract" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ruta/a/tu/factura.jpg"
```

**Ejemplo de uso con Python:**
```python
import requests

url = "http://localhost:8000/extract"
files = {"file": open("factura.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```