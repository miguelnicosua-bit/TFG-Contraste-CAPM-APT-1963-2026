"""
PASO 1 ANUAL - Primera etapa de Fama-MacBeth (frecuencia anual)
TFG Valoración de Activos - Contraste empírico del CAPM
Autor: Miguel Suárez Crespo

Para cada cartera j se estima por MCO el modelo de mercado:

    r_{j,t} - r_{f,t} = alpha_j + beta_{j,M} (r_{M,t} - r_{f,t}) + e_{j,t}

Dos versiones paralelas, replicando el paso 1 mensual:

  (A) ESTÁTICA: T = 62 años, 1 regresión por cartera.
      Sirve para descriptivos y para el test de heterocedasticidad.

  (B) VENTANA MÓVIL: ventana de 10 años, una regresión por cada año del
      periodo muestral usando los 10 años anteriores. La primera ventana
      (1964) emplea los datos auxiliares desde 1954.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
CARPETA_RESULTADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "resultados", "tablas", "capm_anual")
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos     = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_anual.pkl"))
mkt_rf      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf_anual.pkl"))
excesos_ext = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_anual_extendido.pkl"))
mkt_rf_ext  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf_anual_extendido.pkl"))

print(f"Datos cargados:")
print(f"  Periodo muestral (T):    {excesos.shape[0]} años x {excesos.shape[1]} carteras")
print(f"  Periodo extendido:       {excesos_ext.shape[0]} años (incluye 10 años previos)")

VENTANA = 10

# ========================================================================
# (A) BETAS ESTÁTICAS - PARA DESCRIPTIVOS Y TEST DE WHITE
# ========================================================================
resultados = []
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, mkt_rf.rename("Mkt-RF")], axis=1).dropna()
    X = sm.add_constant(datos["Mkt-RF"])
    modelo = sm.OLS(datos[cartera], X).fit()
    resultados.append({
        "cartera":     cartera,
        "alpha":       modelo.params["const"],
        "beta":        modelo.params["Mkt-RF"],
        "alpha_t":     modelo.tvalues["const"],
        "beta_t":      modelo.tvalues["Mkt-RF"],
        "alpha_se":    modelo.bse["const"],
        "r2":          modelo.rsquared,
        "sigma_resid": np.sqrt(modelo.mse_resid),
        "rend_medio":  datos[cartera].mean(),
        "n_obs":       int(modelo.nobs),
    })

res = pd.DataFrame(resultados).set_index("cartera")

print("\n" + "=" * 70)
print("(A) BETAS ESTÁTICAS - DESCRIPTIVOS")
print("=" * 70)
print(f"Beta:       media={res['beta'].mean():.3f}   min={res['beta'].min():.3f}   "
      f"max={res['beta'].max():.3f}   sd={res['beta'].std():.3f}")
print(f"R^2:        media={res['r2'].mean():.3f}   min={res['r2'].min():.3f}   "
      f"max={res['r2'].max():.3f}")
print(f"Alpha medio anual:        {res['alpha'].mean() * 100:.3f}%")
print(f"|Alpha| medio anual:      {res['alpha'].abs().mean() * 100:.3f}%")
print(f"Alfas signif al 5%:       {(res['alpha_t'].abs() > 1.96).sum()}/100")
print(f"Alfas signif al 1%:       {(res['alpha_t'].abs() > 2.58).sum()}/100")

# ========================================================================
# (B) BETAS ROLLING - VENTANA DE 10 AÑOS
# ========================================================================
print("\n" + "=" * 70)
print(f"(B) BETAS CON VENTANA MÓVIL - {VENTANA} años")
print("=" * 70)
print(f"Para cada año t del periodo muestral, se re-estiman las 100 betas")
print(f"con los {VENTANA} años anteriores.")
print(f"\nPrimer año:  {excesos.index[0]}  -> ventana [{excesos_ext.index[0]}, {excesos.index[0] - 1}]")
print(f"Último año:  {excesos.index[-1]} -> ventana [{excesos.index[-1] - VENTANA}, {excesos.index[-1] - 1}]")
print(f"Total de secciones cruzadas: {len(excesos)}")

anios_seccion_cruzada = excesos.index
betas_rolling = pd.DataFrame(index=anios_seccion_cruzada, columns=excesos.columns, dtype=float)
alpha_rolling = pd.DataFrame(index=anios_seccion_cruzada, columns=excesos.columns, dtype=float)
sigma_resid_rolling = pd.DataFrame(index=anios_seccion_cruzada, columns=excesos.columns, dtype=float)

print(f"\nEstimando betas rolling (10 años) ...")
print(f"({len(anios_seccion_cruzada)} ventanas x {excesos.shape[1]} carteras = "
      f"{len(anios_seccion_cruzada) * excesos.shape[1]:,} regresiones)")

for anio_t in anios_seccion_cruzada:
    pos_ext = excesos_ext.index.get_loc(anio_t)
    idx_inicio = pos_ext - VENTANA
    idx_fin    = pos_ext
    anios_ventana = excesos_ext.index[idx_inicio:idx_fin]

    mkt_ventana = mkt_rf_ext.loc[anios_ventana]
    X_ventana = sm.add_constant(mkt_ventana.rename("Mkt-RF"))

    for cartera in excesos_ext.columns:
        y_ventana = excesos_ext.loc[anios_ventana, cartera]
        datos = pd.concat([y_ventana, X_ventana], axis=1).dropna()
        if len(datos) < 6:   # mínimo razonable: 6 obs para 2 parámetros
            continue
        modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
        alpha_rolling.loc[anio_t, cartera] = modelo.params["const"]
        betas_rolling.loc[anio_t, cartera] = modelo.params["Mkt-RF"]
        sigma_resid_rolling.loc[anio_t, cartera] = np.sqrt(modelo.mse_resid)

print("\n" + "-" * 70)
print(f"Betas rolling estimadas: {betas_rolling.shape[0]} años x {betas_rolling.shape[1]} carteras")
print(f"Beta rolling: media={betas_rolling.stack().mean():.3f}   "
      f"min={betas_rolling.stack().min():.3f}   max={betas_rolling.stack().max():.3f}")

# ========================================================================
# GUARDADO
# ========================================================================
res.to_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_anual.pkl"))
res.to_csv(os.path.join(CARPETA_RESULTADOS, "primera_etapa_anual.csv"))

primera_rolling = {
    "betas_rolling":       betas_rolling,
    "alpha_rolling":       alpha_rolling,
    "sigma_resid_rolling": sigma_resid_rolling,
    "ventana":             VENTANA,
}
pd.to_pickle(primera_rolling, os.path.join(CARPETA_PROCESADOS, "primera_etapa_anual_rolling.pkl"))

print(f"\n✓ Betas estáticas guardadas en '{CARPETA_PROCESADOS}/primera_etapa_anual.pkl'")
print(f"✓ Betas rolling guardadas en '{CARPETA_PROCESADOS}/primera_etapa_anual_rolling.pkl'")
print("✓ Listos para ejecutar paso2_anual_seccion_cruzada.py")
