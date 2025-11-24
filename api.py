"""
API REST para extracción de datos de facturas
Recibe una foto de factura y devuelve los datos extraídos
"""
import os
import sys
import tempfile
import traceback
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar el directorio raíz al path para importar infer
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Importar función de predicción
try:
    from model.infer import predict
except ImportError:
    # Fallback: importar directamente si model está en el path
    import importlib.util
    spec = importlib.util.spec_from_file_location("infer", ROOT / "model" / "infer.py")
    infer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer_module)
    predict = infer_module.predict

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Extracción de Facturas",
    description="API para extraer datos de facturas usando OCR e IA",
    version="1.0.0"
)

# Configurar CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Extracción de Facturas",
        "version": "1.0.0",
        "endpoints": {
            "/": "Información de la API",
            "/health": "Estado de salud del servicio",
            "/extract": "POST - Extrae datos de una factura (imagen o PDF)"
        }
    }


@app.get("/health")
async def health():
    """Endpoint de salud para verificar que el servicio está funcionando"""
    return {"status": "ok", "service": "invoice-extraction-api"}


@app.post("/extract")
async def extract_invoice_data(
    file: UploadFile = File(..., description="Imagen o PDF de la factura")
):
    """
    Extrae datos de una factura desde una imagen o PDF
    
    **Parámetros:**
    - file: Archivo de imagen (JPG, PNG) o PDF de la factura
    
    **Respuesta:**
    - campos: Datos extraídos (fecha, nit, razón social, subtotal, IVA, total)
    - validaciones: Validaciones realizadas (NIT, total, IVA)
    - status: Estado del procesamiento (ok, warnings)
    - tiempo_procesamiento: Tiempo de procesamiento en segundos
    """
    # Validar tipo de archivo
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Formatos aceptados: {', '.join(allowed_extensions)}"
        )
    
    # Crear archivo temporal
    temp_file = None
    try:
        logger.info(f"Procesando archivo: {file.filename} (tipo: {file.content_type})")
        
        # Validar que el archivo tenga contenido
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="El archivo debe tener un nombre válido"
            )
        
        # Crear directorio temporal si no existe
        temp_dir = Path(tempfile.gettempdir()) / "invoice_api"
        temp_dir.mkdir(exist_ok=True)
        
        # Guardar archivo temporal
        temp_file = temp_dir / f"temp_{file.filename}"
        
        # Leer y guardar el contenido
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="El archivo está vacío"
            )
        
        with open(temp_file, "wb") as f:
            f.write(content)
        
        logger.info(f"Archivo guardado temporalmente en: {temp_file}")
        
        # Validar que el archivo se guardó correctamente
        if not temp_file.exists() or temp_file.stat().st_size == 0:
            raise HTTPException(
                status_code=500,
                detail="Error al guardar el archivo temporal"
            )
        
        # Procesar con la función predict
        logger.info("Iniciando procesamiento con predict()...")
        try:
            result = predict([str(temp_file)])
        except Exception as predict_error:
            logger.error(f"Error en predict(): {str(predict_error)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error al procesar la imagen con OCR: {str(predict_error)}. Verifique que Tesseract esté instalado."
            )
        
        # Verificar si hubo resultados
        if not result or not result.get('results'):
            logger.warning("No se obtuvieron resultados del procesamiento")
            raise HTTPException(
                status_code=500,
                detail="No se pudo procesar la factura. Verifique que el archivo sea una imagen o PDF válido."
            )
        
        # Obtener el primer resultado (ya que solo procesamos un archivo)
        invoice_data = result['results'][0]
        
        logger.info(f"Procesamiento exitoso. Status: {invoice_data.get('status')}")
        
        # Preparar respuesta
        response = {
            "success": True,
            "campos": invoice_data.get('campos', {}),
            "validaciones": invoice_data.get('validaciones', {}),
            "status": invoice_data.get('status', 'unknown'),
            "tiempo_procesamiento": round(result.get('elapsed_s', 0), 3),
            "warnings": result.get('warnings', [])
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error inesperado: {str(e)}")
        logger.error(error_trace)
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la factura: {str(e)}. Detalles técnicos disponibles en los logs del servidor."
        )
    finally:
        # Limpiar archivo temporal
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.info(f"Archivo temporal eliminado: {temp_file}")
            except Exception as cleanup_error:
                logger.warning(f"Error al eliminar archivo temporal: {cleanup_error}")


if __name__ == "__main__":
    # Configuración del servidor
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )

