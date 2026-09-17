"""
Análisis de pasajeros del Titanic.
Limpieza, preprocesamiento, análisis exploratorio y visualización (sin Machine Learning).

Ejecución (desde la carpeta del proyecto):
    python src/analysis.py
"""

from pathlib import Path

import pandas as pd

# Rutas del proyecto (funcionan desde cualquier carpeta en la que se ejecute el script)
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = CARPETA_PROYECTO / "data" / "train.csv"
CARPETA_RESULTADOS = CARPETA_PROYECTO / "outputs" / "resultados"


def titulo(texto):
    print("\n" + "=" * 60)
    print(texto)
    print("=" * 60)


# ------------------------------------------------------------
# 1. Carga del dataset
# ------------------------------------------------------------
def cargar_datos():
    titulo("1. CARGA DEL DATASET")
    df = pd.read_csv(RUTA_DATOS)
    print(f"Archivo cargado: {RUTA_DATOS.name}")
    print(df.head())
    return df


# ------------------------------------------------------------
# 2. Exploración inicial
# ------------------------------------------------------------
def exploracion_inicial(df):
    titulo("2. EXPLORACIÓN INICIAL")

    print(f"Número de pasajeros: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")

    print("\nVariables disponibles:")
    print(list(df.columns))

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores faltantes por columna:")
    faltantes = pd.DataFrame({
        "faltantes": df.isna().sum(),
        "porcentaje": (df.isna().mean() * 100).round(2),
    })
    print(faltantes[faltantes["faltantes"] > 0])

    print(f"\nRegistros duplicados: {df.duplicated().sum()}")

    print("\nEstadísticas descriptivas (variables numéricas):")
    print(df.describe().round(2))

    print("\nEstadísticas descriptivas (variables de texto):")
    print(df.describe(include="str"))


def main():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    exploracion_inicial(df)


if __name__ == "__main__":
    main()
