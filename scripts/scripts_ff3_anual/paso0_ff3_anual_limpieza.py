"""
PASO 0 ANUAL - Limpieza para FF3 (frecuencia anual)
TFG Valoración de Activos - Contraste empírico del FF3
Autor: Miguel Suárez Crespo

Replica el paso 0 mensual del FF3. Carga los factores anuales SMB, HML,
Mkt-RF y las 100 carteras Value-Weighted anuales.
"""

import os
import pandas as pd
import numpy as np

# Resolución de rutas relativas al script (funciona desde cualquier carpeta)
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos")
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

INICIO_MUESTRAL  = 1964
INICIO_EXTENDIDO = 1954

# Factores anuales (sección a partir de la línea 1205)
factores = pd.read_csv(os.path.join(CARPETA_DATOS, "datos_factores.csv"),
                       skiprows=1205, index_col=0)
factores.index.name = "anio"
factores.index = factores.index.astype(str).str.strip()
factores = factores[factores.index.str.match(r"^\d{4}$")]
factores.index = factores.index.astype(int)
factores = factores.apply(pd.to_numeric, errors="coerce") / 100

# Carteras anuales VW
carteras = pd.read_csv(os.path.join(CARPETA_DATOS, "datos_100_carteras.csv"),
                       skiprows=2419, nrows=100, index_col=0)
carteras.index.name = "anio"
carteras.index = carteras.index.astype(str).str.strip()
carteras = carteras[carteras.index.str.match(r"^\d{4}$")]
carteras.index = carteras.index.astype(int)
carteras = carteras.apply(pd.to_numeric, errors="coerce")
carteras = carteras.replace([-99.99, -999, -100], np.nan) / 100

# Versión extendida (desde 1954)
factores_ext = factores.loc[INICIO_EXTENDIDO:].copy()
carteras_ext = carteras.loc[INICIO_EXTENDIDO:].copy()
anios_ext = factores_ext.index.intersection(carteras_ext.index)
factores_ext = factores_ext.loc[anios_ext]
carteras_ext = carteras_ext.loc[anios_ext]

rf_ext            = factores_ext["RF"]
factores_ff3_ext  = factores_ext[["Mkt-RF", "SMB", "HML"]]
excesos_ext       = carteras_ext.sub(rf_ext, axis=0)

# Versión principal (desde 1964)
factores = factores.loc[INICIO_MUESTRAL:]
carteras = carteras.loc[INICIO_MUESTRAL:]
anios_comunes = factores.index.intersection(carteras.index)
factores = factores.loc[anios_comunes]
carteras = carteras.loc[anios_comunes]

rf            = factores["RF"]
factores_ff3  = factores[["Mkt-RF", "SMB", "HML"]]
excesos       = carteras.sub(rf, axis=0)

print("=" * 70)
print("PASO 0 ANUAL FF3")
print("=" * 70)
print(f"Periodo muestral:   {anios_comunes.min()}-{anios_comunes.max()} ({len(anios_comunes)} años)")
print(f"Periodo extendido:  {anios_ext.min()}-{anios_ext.max()}")
print(f"\nPrimas anuales realizadas (%):")
for col in ["Mkt-RF", "SMB", "HML"]:
    print(f"  {col:8} -> media: {factores_ff3[col].mean()*100:6.2f}%   sd: {factores_ff3[col].std()*100:5.2f}%")

excesos.to_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_anual.pkl"))
factores_ff3.to_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_anual.pkl"))
rf.to_pickle(os.path.join(CARPETA_PROCESADOS, "rf_ff3_anual.pkl"))
excesos_ext.to_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_anual_extendido.pkl"))
factores_ff3_ext.to_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_anual_extendido.pkl"))

print(f"\n✓ Datos guardados. Listo para paso1_ff3_anual_betas.py")
