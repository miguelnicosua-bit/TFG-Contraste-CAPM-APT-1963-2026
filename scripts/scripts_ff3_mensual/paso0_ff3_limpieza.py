"""
PASO 0 FF3 - Carga y limpieza de los datos para el modelo de tres factores
TFG Valoración de Activos - Contraste empírico del FF3
Autor: Miguel Suárez Crespo

Carga los factores de Fama-French (Mkt-RF, SMB, HML, RF) y las 100 carteras
Size x BE/ME ponderadas por capitalización. Calcula los excesos de
rendimiento y guarda dos versiones de los datos:

  (A) Versión principal: período muestral del TFG, julio 1963 - abril 2026
      (T = 754 meses). Es la que alimenta toda la cadena del análisis.
  (B) Versión extendida: desde julio 1958, necesaria para que la primera
      ventana móvil de la primera etapa (correspondiente a julio 1963)
      disponga de los 60 meses previos reales.
"""

import os
import pandas as pd
import numpy as np

CARPETA_DATOS = "datos"
CARPETA_PROCESADOS = "datos_procesados"
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

INICIO_MUESTRAL  = "1963-07"   # Estándar Fama-French (1992)
INICIO_EXTENDIDO = "1958-07"   # 60 meses antes, para el rolling

# ========================================================================
# 1) FACTORES DE FAMA-FRENCH
# ========================================================================
factores = pd.read_csv(os.path.join(CARPETA_DATOS, "datos_factores.csv"),
                       skiprows=4, index_col=0)
factores.index.name = "fecha"
factores.index = factores.index.astype(str).str.strip()
factores = factores[factores.index.str.len() == 6]
factores.index = pd.to_datetime(factores.index, format="%Y%m").to_period("M")
factores = factores.apply(pd.to_numeric, errors="coerce") / 100

# ========================================================================
# 2) 100 CARTERAS SIZE x BOOK-TO-MARKET (Value Weighted Monthly)
# ========================================================================
carteras = pd.read_csv(os.path.join(CARPETA_DATOS, "datos_100_carteras.csv"),
                       skiprows=15, nrows=1198, index_col=0)
carteras.index.name = "fecha"
carteras.index = carteras.index.astype(str).str.strip()
carteras.index = pd.to_datetime(carteras.index, format="%Y%m").to_period("M")
carteras = carteras.apply(pd.to_numeric, errors="coerce")
carteras = carteras.replace([-99.99, -999], np.nan) / 100

# ========================================================================
# 3a) VERSIÓN EXTENDIDA (con los 60 meses previos para el rolling)
# ========================================================================
factores_ext = factores.loc[INICIO_EXTENDIDO:].copy()
carteras_ext = carteras.loc[INICIO_EXTENDIDO:].copy()
fechas_ext = factores_ext.index.intersection(carteras_ext.index)
factores_ext = factores_ext.loc[fechas_ext]
carteras_ext = carteras_ext.loc[fechas_ext]

rf_ext            = factores_ext["RF"]
factores_ff3_ext  = factores_ext[["Mkt-RF", "SMB", "HML"]]
excesos_ext       = carteras_ext.sub(rf_ext, axis=0)

# ========================================================================
# 3b) VERSIÓN PRINCIPAL (período muestral del TFG, desde julio 1963)
# ========================================================================
factores = factores.loc[INICIO_MUESTRAL:]
carteras = carteras.loc[INICIO_MUESTRAL:]
fechas_comunes = factores.index.intersection(carteras.index)
factores = factores.loc[fechas_comunes]
carteras = carteras.loc[fechas_comunes]

rf            = factores["RF"]
factores_ff3  = factores[["Mkt-RF", "SMB", "HML"]]
excesos       = carteras.sub(rf, axis=0)

# ========================================================================
# 4) RESUMEN
# ========================================================================
print("=" * 70)
print("RESUMEN DEL PASO 0 - MODELO DE TRES FACTORES (FF3)")
print("=" * 70)
print(f"Período muestral (TFG):       {fechas_comunes.min()} a {fechas_comunes.max()}")
print(f"Número de meses (T):          {len(fechas_comunes)}")
print(f"Número de carteras (N):       {carteras.shape[1]}")
print(f"\nPeríodo extendido (rolling):  {fechas_ext.min()} a {fechas_ext.max()}")
print(f"Número de meses (extendido):  {len(fechas_ext)}")

print(f"\nDatos faltantes (periodo muestral): "
      f"{excesos.isna().sum().sum()} de {excesos.size} "
      f"({100*excesos.isna().sum().sum()/excesos.size:.2f}%)")
print(f"Carteras con algún NaN: {excesos.isna().any().sum()} de 100")

print(f"\nPrimas realizadas anualizadas de los factores (% anual):")
for col in ["Mkt-RF", "SMB", "HML"]:
    media = factores_ff3[col].mean() * 12 * 100
    vol = factores_ff3[col].std() * np.sqrt(12) * 100
    sharpe = (factores_ff3[col].mean() / factores_ff3[col].std()) * np.sqrt(12)
    print(f"  {col:8} -> Media: {media:6.2f}%   Volatilidad: {vol:5.2f}%   Sharpe: {sharpe:.3f}")

# ========================================================================
# 5) GUARDADO
# ========================================================================
excesos.to_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3.pkl"))
factores_ff3.to_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
rf.to_pickle(os.path.join(CARPETA_PROCESADOS, "rf_ff3.pkl"))
excesos_ext.to_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_extendido.pkl"))
factores_ff3_ext.to_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_extendido.pkl"))

print(f"\n✓ Datos principales FF3 guardados en '{CARPETA_PROCESADOS}/'")
print(f"✓ Datos extendidos FF3 guardados en '{CARPETA_PROCESADOS}/' (para rolling)")
print("✓ Listos para ejecutar paso1_ff3_betas.py")
