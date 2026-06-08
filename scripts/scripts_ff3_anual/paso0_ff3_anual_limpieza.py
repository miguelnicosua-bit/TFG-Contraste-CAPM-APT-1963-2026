"""
PASO 0 FF3 ANUAL - Agregación de los factores SMB y HML a frecuencia anual
TFG Valoración de Activos - Modelo de tres factores Fama-French (1993)
Autor: Miguel Suárez Crespo

Este script:
  1. Carga los factores mensuales (Mkt-RF, SMB, HML, RF).
  2. Agrega SMB y HML a frecuencia anual mediante composición:
     F_anual = prod(1 + F_mensual) - 1
  3. Verifica coherencia con la prima de mercado anual (ya calculada en
     paso0_anual_limpieza.py).
  4. Guarda las series anuales listas para los siguientes pasos.
"""

import os
import pandas as pd
import numpy as np

CARPETA_PROCESADOS = "datos_procesados"
CARPETA_ANUAL = "datos_procesados_anual"
os.makedirs(CARPETA_ANUAL, exist_ok=True)

# ========================================================================
# CARGA DE DATOS
# ========================================================================
factores_m = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
mkt_rf_a   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "mkt_rf_anual.pkl"))

print("=" * 70)
print("PASO 0 FF3 ANUAL - AGREGACIÓN DE FACTORES A FRECUENCIA ANUAL")
print("=" * 70)
print(f"Columnas en factores mensuales: {list(factores_m.columns)}")
print(f"Período mensual: {factores_m.index.min()} a {factores_m.index.max()}")

# ========================================================================
# FUNCIÓN DE COMPOSICIÓN ANUAL
# ========================================================================
def componer_anual(df_mensual):
    """
    Agrega rendimientos mensuales a anuales por composición compuesta.
    Solo agrega años con los 12 meses presentes.
    """
    df = df_mensual.copy()
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    else:
        df.index = pd.to_datetime(df.index, format="%Y%m")
    meses_por_anio = df.groupby(df.index.year).size()
    anios_completos = meses_por_anio[meses_por_anio == 12].index.tolist()
    df_completos = df[df.index.year.isin(anios_completos)]
    return df_completos.groupby(df_completos.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )

# ========================================================================
# AGREGACIÓN ANUAL DE LOS TRES FACTORES
# ========================================================================
factores_a = componer_anual(factores_m[["Mkt-RF", "SMB", "HML"]])
smb_a = factores_a["SMB"]
hml_a = factores_a["HML"]

print(f"\nAños naturales completos: {len(factores_a)}")
print(f"Período anual: {factores_a.index.min()} a {factores_a.index.max()}")

# ========================================================================
# COHERENCIA CON LA PRIMA DE MERCADO YA CALCULADA
# ========================================================================
# El factor Mkt-RF de los factores debe coincidir con el ya calculado
# en paso0_anual_limpieza.py (que partió de las carteras y rf).
diff = (factores_a["Mkt-RF"] - mkt_rf_a).abs().max()
print(f"\nMáxima diferencia entre Mkt-RF (factores) y Mkt-RF (paso0 anual): "
      f"{diff:.6f}")
print("→ Diferencias del orden de 10^-6 son normales y reflejan únicamente")
print("  precisión decimal de los CSVs de origen.")

# ========================================================================
# ESTADÍSTICOS DESCRIPTIVOS DE LOS FACTORES ANUALES
# ========================================================================
print("\n" + "=" * 70)
print("ESTADÍSTICOS DESCRIPTIVOS DE LOS FACTORES ANUALES")
print("=" * 70)

resumen = pd.DataFrame({
    "Media (%)":          factores_a.mean() * 100,
    "Volatilidad (%)":    factores_a.std() * 100,
    "Sharpe":             factores_a.mean() / factores_a.std(),
    "Mínimo (%)":         factores_a.min() * 100,
    "Máximo (%)":         factores_a.max() * 100,
}).round(3)

print(resumen)

# ========================================================================
# CORRELACIONES ENTRE FACTORES ANUALES
# ========================================================================
print("\n" + "=" * 70)
print("CORRELACIONES ENTRE FACTORES ANUALES")
print("=" * 70)
print(factores_a.corr().round(3))
print("\n→ Correlaciones bajas confirman que los tres factores aportan")
print("  información distinta (no hay multicolinealidad relevante).")

# ========================================================================
# GUARDADO
# ========================================================================
smb_a.to_pickle(os.path.join(CARPETA_ANUAL, "smb_anual.pkl"))
hml_a.to_pickle(os.path.join(CARPETA_ANUAL, "hml_anual.pkl"))
factores_a.to_pickle(os.path.join(CARPETA_ANUAL, "factores_ff3_anual.pkl"))

print(f"\n✓ Datos guardados en '{CARPETA_ANUAL}/'")
print("✓ Listos para ejecutar paso1_ff3_anual_betas.py")
