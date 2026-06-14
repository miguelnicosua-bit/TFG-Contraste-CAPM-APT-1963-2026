"""
PASO 1 FF3 - Primera etapa: estimación de las betas del modelo de tres factores
TFG Valoración de Activos - Contraste empírico del FF3
Autor: Miguel Suárez Crespo

Implementación de la primera etapa de Fama y MacBeth (1973) extendida al
modelo de tres factores de Fama y French (1993). Para cada cartera j se
estima por MCO la regresión multifactorial:

    r_{j,t} - r_{f,t} = alpha_j + beta_{j,M} (r_{M,t} - r_{f,t})
                         + beta_{j,SMB} SMB_t + beta_{j,HML} HML_t + e_{j,t}

Dos formatos paralelos:

  (A) VERSIÓN ESTÁTICA (T=754, 1 regresión por cartera):
      Sirve para descriptivos: Tabla 5, heatmaps de las tres betas
      factoriales, descriptivos de alfas FF3 (apéndice), test GRS.

  (B) VERSIÓN CON VENTANA MÓVIL (60 meses, 754 regresiones por cartera):
      Para cada mes t del periodo muestral, regresión con los 60 meses
      anteriores [t-60, t-1] usando los datos extendidos (desde julio 1958).
      Es la que alimenta la segunda etapa.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

CARPETA_PROCESADOS = "datos_procesados"
CARPETA_RESULTADOS = "resultados/tablas/ff3_mensual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# Carga de datos principales y extendidos
excesos       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3.pkl"))
factores_ff3  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
excesos_ext       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_extendido.pkl"))
factores_ff3_ext  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_extendido.pkl"))

print(f"Datos cargados (FF3):")
print(f"  Periodo muestral (T):    {excesos.shape[0]} meses x {excesos.shape[1]} carteras")
print(f"  Periodo extendido:       {excesos_ext.shape[0]} meses (incluye 60 meses previos)")

VENTANA = 60

# ========================================================================
# (A) BETAS ESTÁTICAS - PARA DESCRIPTIVOS Y MAPAS DE CALOR
# ========================================================================
resultados = []
for nombre_cartera in excesos.columns:
    y = excesos[nombre_cartera]
    datos = pd.concat([y, factores_ff3], axis=1).dropna()
    X = sm.add_constant(datos[["Mkt-RF", "SMB", "HML"]])
    modelo = sm.OLS(datos.iloc[:, 0], X).fit()
    resultados.append({
        "cartera":     nombre_cartera,
        "alpha":       modelo.params["const"],
        "beta_MKT":    modelo.params["Mkt-RF"],
        "beta_SMB":    modelo.params["SMB"],
        "beta_HML":    modelo.params["HML"],
        "alpha_t":     modelo.tvalues["const"],
        "beta_MKT_t":  modelo.tvalues["Mkt-RF"],
        "beta_SMB_t":  modelo.tvalues["SMB"],
        "beta_HML_t":  modelo.tvalues["HML"],
        "alpha_se":    modelo.bse["const"],
        "r2":          modelo.rsquared,
        "sigma_residual": np.sqrt(modelo.mse_resid),
        "rend_medio":  datos.iloc[:, 0].mean(),
        "n_obs":       int(modelo.nobs),
    })

res = pd.DataFrame(resultados).set_index("cartera")

# Resumen descriptivo
print("\n" + "=" * 70)
print("(A) BETAS ESTÁTICAS - DISTRIBUCIÓN DE LAS BETAS FACTORIALES")
print("=" * 70)
for k in ["beta_MKT", "beta_SMB", "beta_HML"]:
    print(f"\n{k}:")
    print(f"  Media:    {res[k].mean():.3f}     Mediana: {res[k].median():.3f}")
    print(f"  Mínima:   {res[k].min():.3f}     Máxima:  {res[k].max():.3f}")
    print(f"  Desv. típica: {res[k].std():.3f}")

print("\n" + "=" * 70)
print("ALFAS DE JENSEN FF3 (anualizados, en %)")
print("=" * 70)
print(f"Alpha medio (anualizado):    {res['alpha'].mean() * 12 * 100:.3f}%")
print(f"|Alpha| medio anualizado:    {res['alpha'].abs().mean() * 12 * 100:.3f}%")
print(f"Alfas significativos al 5%:  {(res['alpha_t'].abs() > 1.96).sum()} de 100")
print(f"Alfas significativos al 1%:  {(res['alpha_t'].abs() > 2.58).sum()} de 100")

print("\n" + "=" * 70)
print("R^2 DE LAS REGRESIONES MULTIFACTORIALES")
print("=" * 70)
print(f"R^2 medio:    {res['r2'].mean():.3f}")
print(f"R^2 mediano:  {res['r2'].median():.3f}")
print(f"R^2 mínimo:   {res['r2'].min():.3f}  ({res['r2'].idxmin()})")
print(f"R^2 máximo:   {res['r2'].max():.3f}  ({res['r2'].idxmax()})")

# ========================================================================
# (B) BETAS ROLLING - METODOLOGÍA FAMA Y MACBETH (1973) PARA FF3
# ========================================================================
print("\n" + "=" * 70)
print(f"(B) BETAS CON VENTANA MÓVIL FF3 - {VENTANA} meses")
print("=" * 70)
print(f"Para cada mes t del periodo muestral, se re-estiman las 100 ternas")
print(f"de betas (beta_M, beta_SMB, beta_HML) con los {VENTANA} meses anteriores.")
print(f"\nFechas extremas:")
print(f"  Primer mes:  {excesos.index[0]}  -> ventana [{excesos_ext.index[0]}, {excesos.index[0] - 1}]")
print(f"  Último mes:  {excesos.index[-1]}  -> ventana [{excesos.index[-1] - VENTANA}, {excesos.index[-1] - 1}]")
print(f"\nTotal de secciones cruzadas: {len(excesos)} (todo el periodo muestral).")

fechas_seccion_cruzada = excesos.index

beta_MKT_rolling   = pd.DataFrame(index=fechas_seccion_cruzada, columns=excesos.columns, dtype=float)
beta_SMB_rolling   = pd.DataFrame(index=fechas_seccion_cruzada, columns=excesos.columns, dtype=float)
beta_HML_rolling   = pd.DataFrame(index=fechas_seccion_cruzada, columns=excesos.columns, dtype=float)
alpha_rolling      = pd.DataFrame(index=fechas_seccion_cruzada, columns=excesos.columns, dtype=float)

print(f"\nEstimando betas FF3 con ventana móvil...")
print(f"({len(fechas_seccion_cruzada)} ventanas x {excesos.shape[1]} carteras = "
      f"{len(fechas_seccion_cruzada) * excesos.shape[1]:,} regresiones multifactoriales)")

for fecha_t in fechas_seccion_cruzada:
    pos_ext = excesos_ext.index.get_loc(fecha_t)
    idx_inicio = pos_ext - VENTANA
    idx_fin    = pos_ext
    fechas_ventana = excesos_ext.index[idx_inicio:idx_fin]

    factores_ventana = factores_ff3_ext.loc[fechas_ventana]
    X_ventana = sm.add_constant(factores_ventana)

    for cartera in excesos_ext.columns:
        y_ventana = excesos_ext.loc[fechas_ventana, cartera]
        datos = pd.concat([y_ventana, X_ventana], axis=1).dropna()
        if len(datos) < 30:
            continue
        modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
        alpha_rolling.loc[fecha_t, cartera]     = modelo.params["const"]
        beta_MKT_rolling.loc[fecha_t, cartera]  = modelo.params["Mkt-RF"]
        beta_SMB_rolling.loc[fecha_t, cartera]  = modelo.params["SMB"]
        beta_HML_rolling.loc[fecha_t, cartera]  = modelo.params["HML"]

print("\n" + "-" * 70)
print(f"Betas rolling FF3 estimadas: {beta_MKT_rolling.shape[0]} meses x "
      f"{beta_MKT_rolling.shape[1]} carteras (cada una de las 3 betas)")
print(f"Promedios entre celdas:")
print(f"  beta_MKT media: {beta_MKT_rolling.stack().mean():.3f}")
print(f"  beta_SMB media: {beta_SMB_rolling.stack().mean():.3f}")
print(f"  beta_HML media: {beta_HML_rolling.stack().mean():.3f}")

# ========================================================================
# GUARDADO
# ========================================================================
res.to_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))
res.to_csv(os.path.join(CARPETA_RESULTADOS, "primera_etapa_ff3.csv"))

primera_ff3_rolling = {
    "beta_MKT_rolling": beta_MKT_rolling,
    "beta_SMB_rolling": beta_SMB_rolling,
    "beta_HML_rolling": beta_HML_rolling,
    "alpha_rolling":    alpha_rolling,
    "ventana":          VENTANA,
}
pd.to_pickle(primera_ff3_rolling, os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_rolling.pkl"))

print(f"\n✓ (A) Betas estáticas FF3 guardadas en '{CARPETA_PROCESADOS}/primera_etapa_ff3.pkl'")
print(f"✓ (B) Betas rolling FF3 guardadas en '{CARPETA_PROCESADOS}/primera_etapa_ff3_rolling.pkl'")
print(f"✓ Tabla CSV exportada a '{CARPETA_RESULTADOS}/primera_etapa_ff3.csv'")
print("✓ Listos para ejecutar paso2_ff3_seccion_cruzada.py")
