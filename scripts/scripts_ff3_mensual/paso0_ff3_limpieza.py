"""
PASO 0 FF3 - Verificación de datos para el modelo de 3 factores
TFG Valoración de Activos - Modelo de tres factores Fama-French (1993)
Autor: Miguel Suárez Crespo

Este script:
  1. Reutiliza los datos ya procesados en el Paso 0 del análisis CAPM.
  2. Verifica que los factores SMB y HML están disponibles.
  3. Calcula estadísticos descriptivos de los tres factores (MKT, SMB, HML).
  4. Guarda las series listas para el análisis de tres factores.
"""

import os
import pandas as pd
import numpy as np

# ========================================================================
# CARGA DE DATOS YA PROCESADOS
# ========================================================================
CARPETA_PROCESADOS = "datos_procesados"

excesos  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf   = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))
rf       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "rf.pkl"))
factores = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores.pkl"))

# Verificamos que los factores SMB y HML están en el DataFrame
print("=" * 70)
print("PASO 0 FF3 - VERIFICACIÓN DE DATOS")
print("=" * 70)
print(f"Columnas disponibles en 'factores': {list(factores.columns)}")
print(f"Período: {factores.index.min()} a {factores.index.max()}")
print(f"Número de meses: {len(factores)}")
print(f"Número de carteras: {excesos.shape[1]}")

# Extraemos las series de los tres factores
smb = factores["SMB"]
hml = factores["HML"]

# ========================================================================
# ESTADÍSTICOS DESCRIPTIVOS DE LOS TRES FACTORES
# ========================================================================
print("\n" + "=" * 70)
print("ESTADÍSTICOS DESCRIPTIVOS DE LOS FACTORES (anualizados)")
print("=" * 70)

descrip = pd.DataFrame({
    "Mkt-RF": mkt_rf,
    "SMB":    smb,
    "HML":    hml,
})

resumen = pd.DataFrame({
    "Media (% anual)":        descrip.mean() * 12 * 100,
    "Volatilidad (% anual)":  descrip.std() * np.sqrt(12) * 100,
    "Sharpe (anual)":         (descrip.mean() / descrip.std()) * np.sqrt(12),
    "Mínimo mensual (%)":     descrip.min() * 100,
    "Máximo mensual (%)":     descrip.max() * 100,
}).round(3)

print(resumen)

# ========================================================================
# CORRELACIONES ENTRE FACTORES
# ========================================================================
print("\n" + "=" * 70)
print("CORRELACIONES ENTRE FACTORES")
print("=" * 70)
print(descrip.corr().round(3))

# Idealmente, los factores deberían tener correlaciones bajas entre sí
# para que cada uno aporte información distinta. Correlaciones muy altas
# generarían problemas de multicolinealidad en la primera etapa.

# ========================================================================
# GUARDADO PARA LOS SIGUIENTES PASOS
# ========================================================================
smb.to_pickle(os.path.join(CARPETA_PROCESADOS, "smb.pkl"))
hml.to_pickle(os.path.join(CARPETA_PROCESADOS, "hml.pkl"))

# Guardamos también un DataFrame con los tres factores como matriz X
factores_ff3 = pd.DataFrame({
    "Mkt-RF": mkt_rf,
    "SMB":    smb,
    "HML":    hml,
})
factores_ff3.to_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))

print(f"\n✓ Datos guardados en '{CARPETA_PROCESADOS}/'")
print("✓ Listos para ejecutar paso1_ff3_betas.py")
