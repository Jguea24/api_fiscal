# ============================================================
# API DE PREDICCIÓN DE PRESIÓN FISCAL
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = "modelo_random_forest_presion_fiscal.joblib"
FRONTEND_PATH = "frontend/index.html"


# ============================================================
# VERIFICAR MODELO
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo: {MODEL_PATH}"
    )


# ============================================================
# CARGAR MODELO
# ============================================================

modelo = joblib.load(MODEL_PATH)


# ============================================================
# CREAR APLICACIÓN FASTAPI
# ============================================================

app = FastAPI(
    title="API de Predicción de Presión Fiscal",
    description=(
        "Sistema de predicción de presión fiscal "
        "desarrollado mediante Minería de Datos "
        "y Machine Learning."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELO DE DATOS
# ============================================================

class DatosPrediccion(BaseModel):

    region: str = Field(
        ...,
        description="Región geográfica"
    )

    pais: str = Field(
        ...,
        description="País"
    )

    anio: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Año"
    )

    tipo_presion: str = Field(
        ...,
        description="Tipo de presión fiscal"
    )


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@app.get("/", include_in_schema=False)
def inicio():

    return FileResponse(FRONTEND_PATH)


# ============================================================
# ESTADO DE LA API
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "modelo_cargado": True,
        "modelo": "Random Forest"
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post("/predict")
def predecir(datos: DatosPrediccion):

    # --------------------------------------------------------
    # Crear DataFrame con los datos recibidos
    # --------------------------------------------------------

    entrada = pd.DataFrame([
        {
            "Region": datos.region,
            "País": datos.pais,
            "Año": datos.anio,
            "Tipo_Presion": datos.tipo_presion
        }
    ])

    # --------------------------------------------------------
    # Realizar predicción
    # --------------------------------------------------------

    prediccion = modelo.predict(entrada)[0]

    prediccion = float(prediccion)

    # --------------------------------------------------------
    # Convertir a porcentaje
    # --------------------------------------------------------

    porcentaje = prediccion * 100

    # --------------------------------------------------------
    # Respuesta
    # --------------------------------------------------------

    return {
        "prediccion_presion": prediccion,
        "porcentaje": round(porcentaje, 2),

        "datos": {
            "region": datos.region,
            "pais": datos.pais,
            "anio": datos.anio,
            "tipo_presion": datos.tipo_presion
        }
    }