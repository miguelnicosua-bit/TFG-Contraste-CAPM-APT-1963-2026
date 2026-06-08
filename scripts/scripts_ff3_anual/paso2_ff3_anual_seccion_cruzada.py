"""
PASO 2 FF3 ANUAL - Segunda etapa Fama-MacBeth con tres factores (anual)
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Para cada año t, regresamos los excesos de rendimiento de las 100 carteras
sobre las tres betas estimadas en el Paso 1 FF3 anual:

    r_{j,t} - r_{f,t} = gamma_{0,t}
                      + gamma_{1,t} * beta_j^MKT
                      + gamma_{2,t} * beta_j^SMB
                      + gamma_{3,t} * beta_j^HML
                      + u_{j,t}

Luego promediamos los T = 62 pares y aplicamos Fama-MacBeth.

⚠ NOTA TÉCNICA: el test GRS con T=62 < N+K+1=104 no es factible con las
100 carteras. Se aborda en el paso 3 FF3 anual.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_ANUAL = "datos_procesados_anual"
CARPETA_RESULTADOS = "resultados/tablas/ff3_anual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos        = pd.read_pickle(os.path.join(CARPETA_ANUAL, "excesos_carteras_anual.pkl"))
factores_ff3   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "factores_ff3_anual.pkl"))
primera_ff3_a  = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_ff3_anual.pkl"))

betas = primera_ff3_a[["beta_MKT", "beta_SMB", "beta_HML"]]

print(f"Datos cargados:")
print(f"  - Excesos anuales:        {excesos.shape}")
print(f"  - Betas factoriales:      {betas.shape}")
print(f"  - Factores anuales:       {factores_ff3.shape}")

# ========================================================================
# REGRESIONES DE SECCIÓN CRUZADA AÑO A AÑO (con 3 betas)
# ========================================================================
X_design = sm.add_constant(betas)

gammas = []
fechas_t = []

for anio in excesos.index:
    y = excesos.loc[anio]
    datos = pd.concat([y, X_design], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas.append(modelo.params.values)
    fechas_t.append(anio)

gammas = pd.DataFrame(
    gammas,
    columns=["gamma_0", "gamma_MKT", "gamma_SMB", "gamma_HML"],
    index=fechas_t
)

print(f"\nNúmero de regresiones transversales anuales: {len(gammas)}")

# ========================================================================
# AGREGACIÓN TEMPORAL: MEDIA Y T-ESTADÍSTICOS
# ========================================================================
T = len(gammas)

resumen = pd.DataFrame(index=gammas.columns)
resumen["media"]            = gammas.mean().values
resumen["desv_tipica"]      = gammas.std().values
resumen["error_estandar"]   = resumen["desv_tipica"] / np.sqrt(T)
resumen["t_stat"]           = resumen["media"] / resumen["error_estandar"]
resumen["p_valor"]          = pd.Series(
    2 * (1 - stats.t.cdf(resumen["t_stat"].abs(), df=T-1)),
    index=resumen.index
)
resumen["media_pct"]        = resumen["media"] * 100  # ya en términos anuales

print("\n" + "=" * 70)
print("RESULTADOS DE LA SEGUNDA ETAPA FAMA-MACBETH FF3 (DATOS ANUALES)")
print("=" * 70)
print(resumen.round(5))

# ========================================================================
# COMPARACIÓN CON LAS PRIMAS REALIZADAS DE CADA FACTOR
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE: gamma_k estimada vs prima realizada (FF3 anual)")
print("=" * 70)

primas_realizadas = factores_ff3.mean()

comparacion = pd.DataFrame({
    "Factor":                ["MKT", "SMB", "HML"],
    "gamma estimada (%)": [
        resumen.loc["gamma_MKT", "media_pct"],
        resumen.loc["gamma_SMB", "media_pct"],
        resumen.loc["gamma_HML", "media_pct"],
    ],
    "Prima realizada (%)": [
        primas_realizadas["Mkt-RF"] * 100,
        primas_realizadas["SMB"] * 100,
        primas_realizadas["HML"] * 100,
    ],
    "t-stat de gamma": [
        resumen.loc["gamma_MKT", "t_stat"],
        resumen.loc["gamma_SMB", "t_stat"],
        resumen.loc["gamma_HML", "t_stat"],
    ],
    "p-valor": [
        resumen.loc["gamma_MKT", "p_valor"],
        resumen.loc["gamma_SMB", "p_valor"],
        resumen.loc["gamma_HML", "p_valor"],
    ],
})
print(comparacion.round(4).to_string(index=False))

# ========================================================================
# COMPARACIÓN FF3 MENSUAL vs FF3 ANUAL (SEGUNDA ETAPA)
# ========================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN SEGUNDA ETAPA FF3: MENSUAL vs ANUAL")
print("=" * 70)

segunda_ff3_m = pd.read_pickle(os.path.join("datos_procesados", "segunda_etapa_ff3.pkl"))

comp_freq = pd.DataFrame({
    "Coeficiente": ["gamma_0", "gamma_MKT", "gamma_SMB", "gamma_HML"],
    "FF3 mensual (% anual)": [
        f"{segunda_ff3_m.loc['gamma_0',    'media_anual_%']:.3f}",
        f"{segunda_ff3_m.loc['gamma_MKT',  'media_anual_%']:.3f}",
        f"{segunda_ff3_m.loc['gamma_SMB',  'media_anual_%']:.3f}",
        f"{segunda_ff3_m.loc['gamma_HML',  'media_anual_%']:.3f}",
    ],
    "FF3 anual (% anual)": [
        f"{resumen.loc['gamma_0',   'media_pct']:.3f}",
        f"{resumen.loc['gamma_MKT', 'media_pct']:.3f}",
        f"{resumen.loc['gamma_SMB', 'media_pct']:.3f}",
        f"{resumen.loc['gamma_HML', 'media_pct']:.3f}",
    ],
    "t mensual": [
        f"{segunda_ff3_m.loc['gamma_0',    't_stat']:.3f}",
        f"{segunda_ff3_m.loc['gamma_MKT',  't_stat']:.3f}",
        f"{segunda_ff3_m.loc['gamma_SMB',  't_stat']:.3f}",
        f"{segunda_ff3_m.loc['gamma_HML',  't_stat']:.3f}",
    ],
    "t anual": [
        f"{resumen.loc['gamma_0',   't_stat']:.3f}",
        f"{resumen.loc['gamma_MKT', 't_stat']:.3f}",
        f"{resumen.loc['gamma_SMB', 't_stat']:.3f}",
        f"{resumen.loc['gamma_HML', 't_stat']:.3f}",
    ],
})
print(comp_freq.to_string(index=False))

# ========================================================================
# GUARDADO
# ========================================================================
gammas.to_pickle(os.path.join(CARPETA_ANUAL, "gammas_ff3_anuales.pkl"))
resumen.to_pickle(os.path.join(CARPETA_ANUAL, "segunda_etapa_ff3_anual.pkl"))
resumen.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_ff3_anual.csv"))
comparacion.to_csv(os.path.join(CARPETA_RESULTADOS, "comparacion_gammas_primas_ff3_anual.csv"), index=False)

print(f"\n✓ Resultados guardados en '{CARPETA_ANUAL}/'")
print("✓ Listos para ejecutar paso3_ff3_anual_graficos.py")
