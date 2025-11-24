"""
Script para iniciar la API de extracción de facturas
"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    # Configuración del servidor
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Iniciando API de Extracción de Facturas en http://{host}:{port}")
    print(f"📚 Documentación disponible en http://{host}:{port}/docs")
    print(f"🔍 Health check en http://{host}:{port}/health")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )

