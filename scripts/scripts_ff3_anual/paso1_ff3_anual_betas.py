"""
PASO 1 ANUAL FF3 - Primera etapa multifactorial (frecuencia anual)
Autor: Miguel Suárez Crespo

Para cada cartera j:
    r_{j,t} - r_{f,t} = alpha + beta_M (r_M - r_f) + beta_SMB SMB + beta_HML HML + e

Dos versiones: estática (T=62) y rolling (ventana 10 años).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
CARPETA_RESULTADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "resultados", "tablas", "ff3_anual")
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_anual.pkl"))
factores_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_anual.pkl"))
excesos_ext      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_anual_extendido.pkl"))
factores_ff3_ext = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_anual_extendido.pkl"))

VENTANA = 10

# ========== (A) ESTÁTICA ==========
resultados = []
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, factores_ff3], axis=1).dropna()
    X = sm.add_constant(datos[["Mkt-RF", "SMB", "HML"]])
    modelo = sm.OLS(datos.iloc[:, 0], X).fit()
    resultados.append({
        "cartera":    cartera,
        "alpha":      modelo.params["const"],
        "beta_MKT":   modelo.params["Mkt-RF"],
        "beta_SMB":   modelo.params["SMB"],
        "beta_HML":   modelo.params["HML"],
        "alpha_t":    modelo.tvalues["const"],
        "beta_MKT_t": modelo.tvalues["Mkt-RF"],
        "beta_SMB_t": modelo.tvalues["SMB"],
        "beta_HML_t": modelo.tvalues["HML"],
        "alpha_se":   modelo.bse["const"],
        "r2":         modelo.rsquared,
        "sigma_residual": np.sqrt(modelo.mse_resid),
        "rend_medio": datos.iloc[:, 0].mean(),
    })

res = pd.DataFrame(resultados).set_index("cartera")

print("=" * 70)
print("(A) BETAS ESTÁTICAS FF3 ANUAL")
print("=" * 70)
for k in ["beta_MKT", "beta_SMB", "beta_HML"]:
    print(f"  {k}: media={res[k].mean():.3f}   min={res[k].min():.3f}   max={res[k].max():.3f}")
print(f"  R^2 medio:   {res['r2'].mean():.3f}")
print(f"  |Alpha| medio anual:  {res['alpha'].abs().mean() * 100:.3f}%")
print(f"  Alfas signif 5%:      {(res['alpha_t'].abs() > 1.96).sum()}/100")
print(f"  Alfas signif 1%:      {(res['alpha_t'].abs() > 2.58).sum()}/100")

# ========== (B) ROLLING ==========
print("\n" + "=" * 70)
print(f"(B) BETAS ROLLING FF3 - {VENTANA} años")
print("=" * 70)

anios = excesos.index
beta_MKT_r = pd.DataFrame(index=anios, columns=excesos.columns, dtype=float)
beta_SMB_r = pd.DataFrame(index=anios, columns=excesos.columns, dtype=float)
beta_HML_r = pd.DataFrame(index=anios, columns=excesos.columns, dtype=float)
alpha_r    = pd.DataFrame(index=anios, columns=excesos.columns, dtype=float)

for anio_t in anios:
    pos_ext = excesos_ext.index.get_loc(anio_t)
    anios_ventana = excesos_ext.index[pos_ext - VENTANA : pos_ext]
    factores_ventana = factores_ff3_ext.loc[anios_ventana]
    X_ventana = sm.add_constant(factores_ventana)

    for cartera in excesos_ext.columns:
        y_ventana = excesos_ext.loc[anios_ventana, cartera]
        datos = pd.concat([y_ventana, X_ventana], axis=1).dropna()
        if len(datos) < 6:
            continue
        modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
        alpha_r.loc[anio_t, cartera]    = modelo.params["const"]
        beta_MKT_r.loc[anio_t, cartera] = modelo.params["Mkt-RF"]
        beta_SMB_r.loc[anio_t, cartera] = modelo.params["SMB"]
        beta_HML_r.loc[anio_t, cartera] = modelo.params["HML"]

print(f"\nBetas rolling FF3 estimadas: {beta_MKT_r.shape[0]} años x {beta_MKT_r.shape[1]} carteras")
for nombre, df in [("beta_MKT", beta_MKT_r), ("beta_SMB", beta_SMB_r), ("beta_HML", beta_HML_r)]:
    print(f"  {nombre}: media={df.stack().mean():.3f}   min={df.stack().min():.3f}   max={df.stack().max():.3f}")

# GUARDADO
res.to_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_anual.pkl"))
res.to_csv(os.path.join(CARPETA_RESULTADOS, "primera_etapa_ff3_anual.csv"))

primera_rolling = {
    "beta_MKT_rolling": beta_MKT_r,
    "beta_SMB_rolling": beta_SMB_r,
    "beta_HML_rolling": beta_HML_r,
    "alpha_rolling":    alpha_r,
    "ventana":          VENTANA,
}
pd.to_pickle(primera_rolling, os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_anual_rolling.pkl"))
print("\n✓ Listo para paso2_ff3_anual_seccion_cruzada.py")
