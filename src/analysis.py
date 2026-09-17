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


# ------------------------------------------------------------
# 4. Nuevas variables
# ------------------------------------------------------------
def nuevas_variables(df):
    titulo("4. NUEVAS VARIABLES")
    df = df.copy()

    # FamilySize: hermanos/pareja (SibSp) + padres/hijos (Parch) + el propio pasajero
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # TravelAlone: "Solo" si FamilySize es 1, "Acompañado" si es mayor
    df["TravelAlone"] = df["FamilySize"].apply(lambda x: "Solo" if x == 1 else "Acompañado")

    # AgeGroup: criterios definidos para este proyecto
    #   Niño:          0 a 12 años
    #   Joven:        13 a 29 años
    #   Adulto:       30 a 59 años
    #   Adulto mayor: 60 años o más
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 29, 59, 120],
        labels=["Niño", "Joven", "Adulto", "Adulto mayor"],
        include_lowest=True,
    )

    # FareGroup: la tarifa dividida en 4 grupos con la misma cantidad de pasajeros (cuartiles)
    df["FareGroup"] = pd.qcut(df["Fare"], q=4, labels=["Baja", "Media-baja", "Media-alta", "Alta"])

    print("Variables creadas: FamilySize, TravelAlone, AgeGroup, FareGroup")
    print("\nPasajeros por grupo de edad:")
    print(df["AgeGroup"].value_counts().sort_index())
    print("\nTamaño de familia:")
    print(df["FamilySize"].value_counts().sort_index())
    return df


# ------------------------------------------------------------
# 5. Análisis
# ------------------------------------------------------------
def tasa_supervivencia(df, columna):
    """Tabla con número de pasajeros, sobrevivientes y % de supervivencia por grupo."""
    tabla = df.groupby(columna, observed=True)["Survived"].agg(pasajeros="count", sobrevivientes="sum")
    tabla["supervivencia_%"] = (tabla["sobrevivientes"] / tabla["pasajeros"] * 100).round(1)
    return tabla


def analisis(df):
    titulo("5. ANÁLISIS")
    resultados = {}

    # Pregunta 1
    print("\nP1. ¿Qué porcentaje de pasajeros sobrevivió?")
    total = len(df)
    sobrevivientes = df["Survived"].sum()
    print(f"Sobrevivieron {sobrevivientes} de {total} pasajeros ({sobrevivientes / total * 100:.1f} %).")

    # Pregunta 2
    print("\nP2. ¿Cómo cambia la supervivencia entre hombres y mujeres?")
    resultados["sexo"] = tasa_supervivencia(df, "Sex")
    print(resultados["sexo"])

    # Pregunta 3
    print("\nP3. ¿Cómo cambia la supervivencia según la clase del pasajero?")
    resultados["clase"] = tasa_supervivencia(df, "Pclass")
    print(resultados["clase"])
    print("\nSupervivencia (%) por clase y sexo:")
    print((df.pivot_table(index="Pclass", columns="Sex", values="Survived") * 100).round(1))

    # Pregunta 4
    print("\nP4. ¿Qué grupos de edad presentan mayor supervivencia?")
    resultados["edad"] = tasa_supervivencia(df, "AgeGroup")
    print(resultados["edad"])

    # Pregunta 5
    print("\nP5. ¿Viajar solo o acompañado está relacionado con la supervivencia?")
    resultados["solo"] = tasa_supervivencia(df, "TravelAlone")
    print(resultados["solo"])
    print("\nPor tamaño de familia:")
    resultados["familia"] = tasa_supervivencia(df, "FamilySize")
    print(resultados["familia"])

    # Pregunta 6
    print("\nP6. ¿Existe relación entre la tarifa pagada y la supervivencia?")
    resultados["tarifa"] = tasa_supervivencia(df, "FareGroup")
    print(resultados["tarifa"])
    print("\nTarifa mediana según supervivencia:")
    print(df.groupby("Survived_texto")["Fare"].median().round(2))

    # Guardar las tablas en CSV
    for nombre, tabla in resultados.items():
        tabla.to_csv(CARPETA_RESULTADOS / f"supervivencia_por_{nombre}.csv")
    print("\nTablas guardadas en outputs/resultados/")
    return resultados


def main():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    exploracion_inicial(df)
    df = limpieza(df)
    df = nuevas_variables(df)
    analisis(df)


if __name__ == "__main__":
    main()
