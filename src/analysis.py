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


# ------------------------------------------------------------
# 3. Limpieza y preprocesamiento
# ------------------------------------------------------------
def limpieza(df):
    titulo("3. LIMPIEZA Y PREPROCESAMIENTO")
    df = df.copy()

    # Age (177 faltantes, 19.9 %):
    # No se eliminan esas filas porque se perdería casi una quinta parte de los datos.
    # Se rellenan con la mediana de edad de los pasajeros del mismo sexo y clase,
    # ya que la edad cambia bastante entre clases (en 1ra clase son mayores).
    # Se usa la mediana y no el promedio porque no la afectan tanto los valores extremos.
    mediana_edad = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(mediana_edad)
    print("Age: faltantes rellenados con la mediana por sexo y clase.")

    # Cabin (687 faltantes, 77.1 %):
    # Faltan demasiados datos para rellenarlos sin inventar información.
    # Se crea la variable HasCabin (1 = tiene cabina registrada, 0 = no)
    # y se elimina la columna Cabin original.
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df = df.drop(columns=["Cabin"])
    print("Cabin: se reemplaza por HasCabin (1/0) y se elimina la columna.")

    # Embarked (2 faltantes, 0.2 %):
    # Son muy pocos, se rellenan con el puerto más frecuente (la moda), que es S (Southampton).
    moda_embarque = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(moda_embarque)
    print(f"Embarked: faltantes rellenados con la moda ({moda_embarque}).")

    # Transformaciones para que los resultados sean más fáciles de leer
    df["Sex"] = df["Sex"].map({"male": "Hombre", "female": "Mujer"})
    df["Embarked"] = df["Embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
    df["Survived_texto"] = df["Survived"].map({0: "No sobrevivió", 1: "Sobrevivió"})

    # Ticket no se usa en el análisis
    df = df.drop(columns=["Ticket"])

    print(f"\nFaltantes después de la limpieza: {df.isna().sum().sum()}")
    print(f"Duplicados después de la limpieza: {df.duplicated().sum()}")

    ruta = CARPETA_RESULTADOS / "titanic_limpio.csv"
    df.to_csv(ruta, index=False)
    print(f"Dataset limpio guardado en: outputs/resultados/{ruta.name}")
    return df


def main():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    exploracion_inicial(df)
    df = limpieza(df)


if __name__ == "__main__":
    main()
