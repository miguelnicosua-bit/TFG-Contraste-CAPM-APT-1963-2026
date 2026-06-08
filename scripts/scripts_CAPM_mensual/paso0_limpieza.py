"""
PASO 0 - Carga y limpieza de los datos de Fama-French
TFG Valoración de Activos - Contraste empírico del CAPM
Autor: Miguel Suárez Crespo
 
Este script:
  1. Carga el archivo de los 3 factores de Fama-French (Mkt-RF, SMB, HML, RF).
  2. Carga la tabla mensual Value-Weighted de las 100 carteras Size x BE/ME.
  3. Filtra ambos al período muestral (desde julio 1963, estándar Fama-French).
  4. Calcula los excesos de rendimiento de cada cartera (r_i - r_f).
  5. Guarda los DataFrames procesados para los pasos siguientes.
"""
 
import os
import pandas as pd
import numpy as np
 
# ========================================================================
# CONFIGURACIÓN DE RUTAS
# ========================================================================
# Trabajamos con rutas relativas a la carpeta del proyecto.
# Esto hace el código portable: funciona en cualquier ordenador.
 
CARPETA_DATOS = "datos"
CARPETA_PROCESADOS = "datos_procesados"
 
# Creamos la carpeta de salida si no existe
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

INICIO_MUESTRAL = "1963-07"   # Estándar Fama-French (1992)
 
# ========================================================================
# 1) FACTORES DE FAMA-FRENCH
# ========================================================================
ruta_factores = os.path.join(CARPETA_DATOS, "datos_factores.csv")
 
# Saltamos las 4 primeras líneas (3 de texto descriptivo + 1 en blanco)
factores = pd.read_csv(ruta_factores, skiprows=4, index_col=0)
factores.index.name = "fecha"
 
# Al final del archivo hay datos anuales pegados (formato YYYY = 4 dígitos).
# Filtramos solo las filas con formato mensual YYYYMM (6 dígitos).
factores.index = factores.index.astype(str).str.strip()
factores = factores[factores.index.str.len() == 6]
 
# Convertimos el índice a periodo mensual
factores.index = pd.to_datetime(factores.index, format="%Y%m").to_period("M")
 
# Pasamos de porcentaje a decimales
factores = factores.apply(pd.to_numeric, errors="coerce") / 100
 
# ========================================================================
# 2) 100 CARTERAS SIZE x BOOK-TO-MARKET (Value Weighted Monthly)
# ========================================================================
ruta_carteras = os.path.join(CARPETA_DATOS, "datos_100_carteras.csv")
 
# La tabla VW Monthly empieza en la línea 16 (cabecera) y tiene 1198 filas.
# Las otras 5 tablas del archivo (EW Monthly, VW Annual, etc.) las ignoramos.
carteras = pd.read_csv(ruta_carteras, skiprows=15, nrows=1198, index_col=0)
carteras.index.name = "fecha"
 
# Mismo tratamiento de fechas que con factores
carteras.index = carteras.index.astype(str).str.strip()
carteras.index = pd.to_datetime(carteras.index, format="%Y%m").to_period("M")
 
# A numérico y datos faltantes (-99.99 y -999) como NaN
carteras = carteras.apply(pd.to_numeric, errors="coerce")
carteras = carteras.replace([-99.99, -999], np.nan)
carteras = carteras / 100
 
# ========================================================================
# 3) FILTRO POR PERÍODO MUESTRAL
# ========================================================================
factores = factores.loc[INICIO_MUESTRAL:]
carteras = carteras.loc[INICIO_MUESTRAL:]
 
# Aseguramos que ambos DataFrames tienen exactamente las mismas fechas
fechas_comunes = factores.index.intersection(carteras.index)
factores = factores.loc[fechas_comunes]
carteras = carteras.loc[fechas_comunes]
 
# ========================================================================
# 4) EXCESOS DE RENDIMIENTO (r_i - r_f)
# ========================================================================
rf = factores["RF"]
mkt_rf = factores["Mkt-RF"]
excesos = carteras.sub(rf, axis=0)
 
# ========================================================================
# 5) RESUMEN
# ========================================================================
print("=" * 70)
print("RESUMEN DEL PASO 0")
print("=" * 70)
print(f"Período muestral: {fechas_comunes.min()} a {fechas_comunes.max()}")
print(f"Número de meses (T): {len(fechas_comunes)}")
print(f"Número de carteras (N): {carteras.shape[1]}")
print(f"\nDatos faltantes en carteras: "
      f"{excesos.isna().sum().sum()} de {excesos.size} "
      f"({100*excesos.isna().sum().sum()/excesos.size:.2f}%)")
print(f"Carteras con algún NaN: {excesos.isna().any().sum()} de 100")
 
print(f"\nMedia anualizada del exceso de mercado (Mkt-RF): "
      f"{mkt_rf.mean() * 12 * 100:.2f}%")
print(f"Volatilidad anualizada del exceso de mercado: "
      f"{mkt_rf.std() * np.sqrt(12) * 100:.2f}%")
print(f"Sharpe ratio anualizado del mercado: "
      f"{(mkt_rf.mean() / mkt_rf.std()) * np.sqrt(12):.3f}")
 
# ========================================================================
# 6) GUARDADO PARA LOS SIGUIENTES PASOS
# ========================================================================
excesos.to_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf.to_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))
rf.to_pickle(os.path.join(CARPETA_PROCESADOS, "rf.pkl"))
factores.to_pickle(os.path.join(CARPETA_PROCESADOS, "factores.pkl"))
 
print(f"\n✓ Datos guardados en '{CARPETA_PROCESADOS}/'")
print("✓ Listos para ejecutar paso1_betas.py")