"""
PASO 0 ANUAL - Carga y limpieza de los datos de Fama-French (frecuencia anual)
TFG Valoración de Activos - Contraste empírico del CAPM (replicación anual)
Autor: Miguel Suárez Crespo

Versión anual del análisis empírico. Replica la estructura del paso 0 mensual:
  1. Carga de los 3 factores de Fama-French anuales (Mkt-RF, SMB, HML, RF).
  2. Carga de las 100 carteras Size x BE/ME anuales (Value-Weighted).
  3. Cálculo de los excesos de rendimiento anuales (r_i - r_f).
  4. Versión PRINCIPAL: periodo muestral 1964-2025 (T=62 años).
  5. Versión EXTENDIDA: retrocede 10 años antes del inicio muestral
     (desde 1954), necesaria para estimar las betas rolling de los primeros
     años del periodo muestral.

Decisión metodológica: ventana móvil de 10 años (paralelismo conceptual con
los 60 meses de la versión mensual). 5 años sería insuficiente para el FF3
(necesita estimar 4 parámetros: constante + 3 factores).
"""

import os
import pandas as pd
import numpy as np

# ========================================================================
# CONFIGURACIÓN
# ========================================================================
# Resolución de rutas relativas al script (funciona desde cualquier carpeta)
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos")
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

INICIO_MUESTRAL  = 1964   # Primer año completo siguiente a julio 1963
INICIO_EXTENDIDO = 1954   # 10 años antes, para el rolling de 10 años

# ========================================================================
# 1) FACTORES DE FAMA-FRENCH (sección anual)
# ========================================================================
ruta_factores = os.path.join(CARPETA_DATOS, "datos_factores.csv")

# La sección anual empieza en la línea 1205 del archivo
# (después del header "Annual Factors: January-December")
factores = pd.read_csv(ruta_factores, skiprows=1205, index_col=0)
factores.index.name = "anio"

# Filtramos solo filas con formato YYYY (4 dígitos) que sean válidas
factores.index = factores.index.astype(str).str.strip()
factores = factores[factores.index.str.match(r"^\d{4}$")]
factores.index = factores.index.astype(int)
factores = factores.apply(pd.to_numeric, errors="coerce") / 100

# ========================================================================
# 2) 100 CARTERAS SIZE x BOOK-TO-MARKET (Value Weighted Annual)
# ========================================================================
ruta_carteras = os.path.join(CARPETA_DATOS, "datos_100_carteras.csv")

# La sección anual Value Weighted empieza en la línea 2419
# Termina antes de la sección Equal Weighted que empieza en línea 2522
carteras = pd.read_csv(ruta_carteras, skiprows=2419, nrows=100, index_col=0)
carteras.index.name = "anio"
carteras.index = carteras.index.astype(str).str.strip()
carteras = carteras[carteras.index.str.match(r"^\d{4}$")]
carteras.index = carteras.index.astype(int)
carteras = carteras.apply(pd.to_numeric, errors="coerce")
carteras = carteras.replace([-99.99, -999, -100], np.nan) / 100

# ========================================================================
# 3a) VERSIÓN EXTENDIDA (con los 10 años previos para el rolling)
# ========================================================================
factores_ext = factores.loc[INICIO_EXTENDIDO:].copy()
carteras_ext = carteras.loc[INICIO_EXTENDIDO:].copy()
anios_ext = factores_ext.index.intersection(carteras_ext.index)
factores_ext = factores_ext.loc[anios_ext]
carteras_ext = carteras_ext.loc[anios_ext]

rf_ext     = factores_ext["RF"]
mkt_rf_ext = factores_ext["Mkt-RF"]
excesos_ext = carteras_ext.sub(rf_ext, axis=0)

# ========================================================================
# 3b) VERSIÓN PRINCIPAL (periodo muestral, desde 1964)
# ========================================================================
factores = factores.loc[INICIO_MUESTRAL:]
carteras = carteras.loc[INICIO_MUESTRAL:]
anios_comunes = factores.index.intersection(carteras.index)
factores = factores.loc[anios_comunes]
carteras = carteras.loc[anios_comunes]

rf     = factores["RF"]
mkt_rf = factores["Mkt-RF"]
excesos = carteras.sub(rf, axis=0)

# ========================================================================
# 4) RESUMEN
# ========================================================================
print("=" * 70)
print("RESUMEN DEL PASO 0 ANUAL - CAPM")
print("=" * 70)
print(f"Período muestral (TFG):       {anios_comunes.min()} a {anios_comunes.max()}")
print(f"Número de años (T):           {len(anios_comunes)}")
print(f"Número de carteras (N):       {carteras.shape[1]}")
print(f"\nPeriodo extendido (rolling):  {anios_ext.min()} a {anios_ext.max()}")
print(f"Número de años (extendido):   {len(anios_ext)}")

print(f"\nDatos faltantes (periodo muestral): "
      f"{excesos.isna().sum().sum()} de {excesos.size} "
      f"({100*excesos.isna().sum().sum()/excesos.size:.2f}%)")
print(f"Carteras con algún NaN: {excesos.isna().any().sum()} de 100")

print(f"\nMedia anual del exceso de Mkt-RF:   {mkt_rf.mean()*100:.3f}%")
print(f"Mediana anual:                       {mkt_rf.median()*100:.3f}%")
print(f"Desv. típica anual:                  {mkt_rf.std()*100:.3f}%")

# ========================================================================
# 5) GUARDADO
# ========================================================================
excesos.to_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_anual.pkl"))
mkt_rf.to_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf_anual.pkl"))
rf.to_pickle(os.path.join(CARPETA_PROCESADOS, "rf_anual.pkl"))
excesos_ext.to_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_anual_extendido.pkl"))
mkt_rf_ext.to_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf_anual_extendido.pkl"))

print(f"\n✓ Datos anuales guardados en '{CARPETA_PROCESADOS}/'")
print("✓ Listos para ejecutar paso1_anual_betas.py")
