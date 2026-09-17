"""
Análisis de pasajeros del Titanic.

Limpieza, preprocesamiento, análisis exploratorio y visualización
de los datos del Titanic. Este proyecto no utiliza Machine Learning.

Ejecución desde la carpeta principal del proyecto:
    python src/analysis.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "resultados"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n=== EXPLORACIÓN INICIAL ===")
print(f"Registros: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")
print("\nTipos de datos:")
print(df.dtypes)

print("\nValores faltantes:")
print(df.isnull().sum())

print(f"\nDuplicados: {df.duplicated().sum()}")


# ============================================================
# 2. LIMPIEZA DE DATOS
# ============================================================

# Edad: mediana según sexo y clase
df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Cabina: convertir en indicador de si el pasajero tenía cabina registrada
df["HasCabin"] = df["Cabin"].notna().astype(int)

# Puerto de embarque: completar valores faltantes con el más frecuente
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# ============================================================
# 3. CREACIÓN DE VARIABLES
# ============================================================

# Tamaño de la familia
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Si viajaba solo o acompañado
df["TravelAlone"] = df["FamilySize"].apply(
    lambda x: "Solo" if x == 1 else "Acompañado"
)

# Grupo de edad
def age_group(age):
    if age <= 12:
        return "Niño"
    elif age <= 29:
        return "Joven"
    elif age <= 59:
        return "Adulto"
    else:
        return "Adulto mayor"


df["AgeGroup"] = df["Age"].apply(age_group)

# Grupos de tarifa mediante cuartiles
df["FareGroup"] = pd.qcut(
    df["Fare"],
    q=4,
    labels=["Baja", "Media-baja", "Media-alta", "Alta"]
)


# ============================================================
# 4. ANÁLISIS
# ============================================================

survival_rate = df["Survived"].mean() * 100

survival_by_sex = df.groupby("Sex")["Survived"].mean() * 100
survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100
survival_by_age = df.groupby("AgeGroup", observed=True)["Survived"].mean() * 100
survival_by_travel = df.groupby("TravelAlone")["Survived"].mean() * 100
survival_by_fare = df.groupby("FareGroup", observed=True)["Survived"].mean() * 100

print("\n=== RESULTADOS ===")
print(f"\nSupervivencia general: {survival_rate:.1f}%")

print("\nSupervivencia por sexo:")
print(survival_by_sex.round(1))

print("\nSupervivencia por clase:")
print(survival_by_class.round(1))

print("\nSupervivencia por grupo de edad:")
print(survival_by_age.round(1))

print("\nSupervivencia según si viajaba solo:")
print(survival_by_travel.round(1))

print("\nSupervivencia por grupo de tarifa:")
print(survival_by_fare.round(1))


# ============================================================
# 5. GUARDAR DATASET LIMPIO Y TABLAS
# ============================================================

df.to_csv(OUTPUT_DIR / "titanic_limpio.csv", index=False)

survival_by_sex.to_csv(OUTPUT_DIR / "supervivencia_por_sexo.csv")
survival_by_class.to_csv(OUTPUT_DIR / "supervivencia_por_clase.csv")
survival_by_age.to_csv(OUTPUT_DIR / "supervivencia_por_edad.csv")
survival_by_travel.to_csv(OUTPUT_DIR / "supervivencia_por_viaje.csv")
survival_by_fare.to_csv(OUTPUT_DIR / "supervivencia_por_tarifa.csv")


# ============================================================
# 6. VISUALIZACIONES
# ============================================================

# Supervivencia por sexo
plt.figure(figsize=(7, 5))
survival_by_sex.plot(kind="bar")
plt.title("Supervivencia por sexo")
plt.xlabel("Sexo")
plt.ylabel("Supervivencia (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "supervivencia_por_sexo.png")
plt.close()


# Supervivencia por clase
plt.figure(figsize=(7, 5))
survival_by_class.plot(kind="bar")
plt.title("Supervivencia por clase")
plt.xlabel("Clase")
plt.ylabel("Supervivencia (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "supervivencia_por_clase.png")
plt.close()


# Supervivencia por grupo de edad
age_order = ["Niño", "Joven", "Adulto", "Adulto mayor"]

plt.figure(figsize=(8, 5))
survival_by_age.reindex(age_order).plot(kind="bar")
plt.title("Supervivencia por grupo de edad")
plt.xlabel("Grupo de edad")
plt.ylabel("Supervivencia (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "supervivencia_por_edad.png")
plt.close()


# Supervivencia por tamaño de familia
family_survival = df.groupby("FamilySize")["Survived"].mean() * 100

plt.figure(figsize=(8, 5))
family_survival.plot(kind="bar")
plt.title("Supervivencia según tamaño de familia")
plt.xlabel("Tamaño de familia")
plt.ylabel("Supervivencia (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "supervivencia_por_familia.png")
plt.close()


# Tarifa según supervivencia
plt.figure(figsize=(7, 5))
df.boxplot(column="Fare", by="Survived")
plt.suptitle("")
plt.title("Tarifa según supervivencia")
plt.xlabel("Supervivió (0 = No, 1 = Sí)")
plt.ylabel("Tarifa")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tarifa_por_supervivencia.png")
plt.close()


# ============================================================
# 7. CONCLUSIONES
# ============================================================

conclusiones = f"""
CONCLUSIONES DEL ANÁLISIS DEL TITANIC

Supervivencia general:
El {survival_rate:.1f}% de los pasajeros incluidos en el dataset sobrevivió.

Sexo:
Las mujeres presentaron una tasa de supervivencia considerablemente mayor
que los hombres.

Clase:
La supervivencia fue mayor entre los pasajeros de primera clase, seguida
por segunda clase y finalmente tercera clase.

Edad:
Los niños presentaron una mayor tasa de supervivencia que varios de los
grupos de mayor edad. Los adultos mayores presentaron una de las tasas
más bajas.

Familia:
Los pasajeros que viajaban acompañados presentaron una mayor supervivencia
que quienes viajaban solos. Sin embargo, los grupos familiares muy grandes
presentaron tasas de supervivencia bajas.

Tarifa:
Se observa una relación entre la tarifa pagada y la supervivencia: los
pasajeros de grupos de tarifa más alta presentaron mayores tasas de
supervivencia.

En conjunto, el análisis exploratorio muestra que el sexo, la clase,
la edad, el tamaño de la familia y la tarifa presentan diferencias en las
tasas de supervivencia observadas en este dataset.

Estos resultados describen asociaciones dentro de los datos y no implican
por sí mismos una relación causal.
"""

with open(OUTPUT_DIR / "conclusiones.txt", "w", encoding="utf-8") as file:
    file.write(conclusiones)

print("\nAnálisis terminado correctamente.")
print(f"Resultados guardados en: {OUTPUT_DIR}")
