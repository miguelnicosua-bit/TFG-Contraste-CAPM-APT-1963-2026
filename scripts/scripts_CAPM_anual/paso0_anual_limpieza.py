"""
PASO 0 ANUAL - Limpieza y agregación de datos a frecuencia anual
TFG Valoración de Activos - CAPM con datos anuales
Autor: Miguel Suárez Crespo

Este script:
  1. Carga los datos mensuales ya limpios del Paso 0 original.
  2. Agrega los rendimientos a frecuencia anual mediante composición:
     R_anual = prod(1 + r_mensual) - 1
  3. Filtra años completos (1964-2025): se descartan 1963 (solo jul-dic) y
     2026 (solo ene-abr) por no ser años naturales completos.
  4. Calcula los excesos de rendimiento anuales.
  5. Guarda los datos procesados para los siguientes pasos.
"""

import os
import pandas as pd
import numpy as np

# ========================================================================
# CARGA DE DATOS MENSUALES YA PROCESADOS
# ========================================================================
CARPETA_PROCESADOS = "datos_procesados"

# Para no mezclar con los pickles mensuales, guardamos los anuales aparte
CARPETA_ANUAL = "datos_procesados_anual"
os.makedirs(CARPETA_ANUAL, exist_ok=True)

# Cargamos rendimientos mensuales en bruto (no excesos): necesitamos ambos
# componentes por separado para componerlos correctamente.
# Reconstruimos los rendimientos brutos sumando excesos + rf.
excesos_m = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
rf_m      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "rf.pkl"))
mkt_rf_m  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))

# Rendimientos brutos de las carteras (necesarios para composición correcta)
rend_brutos_m = excesos_m.add(rf_m, axis=0)
# Rendimiento bruto del mercado
rend_mkt_m = mkt_rf_m + rf_m

print("=" * 70)
print("PASO 0 ANUAL - AGREGACIÓN DE DATOS A FRECUENCIA ANUAL")
print("=" * 70)
print(f"Datos mensuales: {len(excesos_m)} meses ({excesos_m.index.min()} a "
      f"{excesos_m.index.max()})")

# ========================================================================
# FUNCIÓN DE COMPOSICIÓN ANUAL
# ========================================================================
def componer_anual(df_mensual):
    """
    Agrega rendimientos mensuales a anuales mediante composición compuesta:
    R_anual = prod(1 + r_mensual) - 1.
    Solo agrega años con los 12 meses presentes (años naturales completos).
    """
    df = df_mensual.copy()
    # El índice puede ser Period o string %Y%m
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    else:
        df.index = pd.to_datetime(df.index, format="%Y%m")
    # Cuenta cuántos meses tiene cada año
    meses_por_anio = df.groupby(df.index.year).size()
    anios_completos = meses_por_anio[meses_por_anio == 12].index.tolist()
    df_completos = df[df.index.year.isin(anios_completos)]
    # Composición compuesta
    return df_completos.groupby(df_completos.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )

# ========================================================================
# AGREGACIÓN ANUAL
# ========================================================================
rend_brutos_a = componer_anual(rend_brutos_m)
rend_mkt_a    = componer_anual(rend_mkt_m.to_frame("Mkt"))["Mkt"]
rf_a          = componer_anual(rf_m.to_frame("RF"))["RF"]

# Filtramos años completos
print(f"\nAños naturales completos identificados: {len(rend_brutos_a)}")
print(f"Período anual: {rend_brutos_a.index.min()} a {rend_brutos_a.index.max()}")

# ========================================================================
# CÁLCULO DE EXCESOS ANUALES
# ========================================================================
excesos_a = rend_brutos_a.sub(rf_a, axis=0)
mkt_rf_a  = rend_mkt_a - rf_a

# ========================================================================
# ESTADÍSTICOS DESCRIPTIVOS BÁSICOS
# ========================================================================
print("\n" + "=" * 70)
print("ESTADÍSTICOS DESCRIPTIVOS DE LOS FACTORES ANUALES")
print("=" * 70)
print(f"Prima de mercado media anual:   {mkt_rf_a.mean() * 100:.3f}%")
print(f"Volatilidad de Mkt-RF (anual):  {mkt_rf_a.std() * 100:.3f}%")
print(f"Sharpe de Mkt-RF:               {mkt_rf_a.mean() / mkt_rf_a.std():.3f}")
print(f"Tipo libre de riesgo medio:     {rf_a.mean() * 100:.3f}%")

print(f"\nN (carteras):                   {excesos_a.shape[1]}")
print(f"T (años):                       {excesos_a.shape[0]}")
print(f"Datos faltantes (%):            {excesos_a.isna().sum().sum() / excesos_a.size * 100:.3f}%")

print("\n⚠ ADVERTENCIA TÉCNICA:")
print(f"  Como T = {excesos_a.shape[0]} < N = {excesos_a.shape[1]}, el test GRS clásico")
print(f"  no es factible con datos anuales (matriz Σ no invertible).")
print(f"  El paso 3 anual lo gestionará adecuadamente.")

# ========================================================================
# GUARDADO
# ========================================================================
excesos_a.to_pickle(os.path.join(CARPETA_ANUAL, "excesos_carteras_anual.pkl"))
mkt_rf_a.to_pickle(os.path.join(CARPETA_ANUAL, "mkt_rf_anual.pkl"))
rf_a.to_pickle(os.path.join(CARPETA_ANUAL, "rf_anual.pkl"))

print(f"\n✓ Datos guardados en '{CARPETA_ANUAL}/'")
print("✓ Listos para ejecutar paso1_anual_betas.py")
