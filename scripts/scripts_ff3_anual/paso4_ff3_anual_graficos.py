"""
PASO 4 ANUAL FF3 - Gráficos del modelo de tres factores (datos anuales)
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Réplica anual del paso4_ff3_graficos.py mensual. Seis figuras SVG:
  1. LMA empírica vs teórica (rendimiento medio vs beta_MKT FF3)
  2. Mapa de calor 10x10 de las betas de mercado (beta_MKT)
  3. Mapa de calor 10x10 de las betas SMB
  4. Mapa de calor 10x10 de las betas HML
  5. Mapa de calor 10x10 de los alfas FF3 (anuales)
  6. Distribución de los t-estadísticos de los alfas FF3
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.colors import TwoSlopeNorm

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
CARPETA_GRAFICOS = os.path.join(CARPETA_SCRIPT, "..", "..", "resultados", "graficos", "ff3_anual")
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

primera      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_anual.pkl"))
factores_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_anual.pkl"))
segunda      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3_anual.pkl"))

g0_anual_ff3   = segunda.loc["gamma_0",   "media"] * 100
gMKT_anual_ff3 = segunda.loc["gamma_MKT", "media"] * 100

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 100,
})

def parsear_nombre(nombre):
    n = nombre.strip()
    if n == "SMALL LoBM": return 1, 1
    if n == "SMALL HiBM": return 1, 10
    if n == "BIG LoBM":   return 10, 1
    if n == "BIG HiBM":   return 10, 10
    m = re.match(r"ME(\d+)\s+BM(\d+)", n)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Nombre no reconocido: {nombre}")

# Calcular rendimiento medio anual por cartera (no viene en primera_etapa_ff3_anual)
excesos = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_anual.pkl"))
rend_medio = excesos.mean()

parsed = pd.DataFrame(
    [parsear_nombre(c) for c in primera.index],
    columns=["size", "bm"],
    index=primera.index
)
datos = pd.concat([primera, parsed], axis=1)
datos["rend_medio"] = rend_medio

# ---- Gráfico 1: LMA ----
fig, ax = plt.subplots(figsize=(9, 6))
betas = datos["beta_MKT"]
rend_anual_scatter = datos["rend_medio"] * 100
prima_mkt = factores_ff3["Mkt-RF"].mean() * 100

sc = ax.scatter(betas, rend_anual_scatter, c=datos["bm"], cmap="RdYlGn",
                s=70, edgecolors="black", linewidths=0.4, alpha=0.85, zorder=3)
x_teo = np.linspace(0.5, 1.7, 100)
ax.plot(x_teo, x_teo * prima_mkt, "--", color="black", linewidth=2,
        label=f"LMA teórica del CAPM (pendiente = {prima_mkt:.2f}%)", zorder=2)
ax.plot(x_teo, g0_anual_ff3 + gMKT_anual_ff3 * x_teo, "-", color="red", linewidth=2,
        label=f"LMA empírica FF3 (pendiente = {gMKT_anual_ff3:.2f}%)", zorder=2)
ax.scatter([1.0], [prima_mkt], marker="*", s=300, color="gold",
           edgecolors="black", linewidths=1.0, zorder=4, label="Cartera de mercado")
ax.set_xlabel("Beta de mercado FF3 ($\\hat{\\beta}_{i,MKT}$)")
ax.set_ylabel("Exceso de rendimiento medio anual (%)")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, ticks=[1, 5, 10])
cbar.set_label("Decil de Book-to-Market\n(1 = growth, 10 = value)",
               rotation=270, labelpad=35)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "01_LMA_teorica_vs_empirica_FF3_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 1: 01_LMA_teorica_vs_empirica_FF3_anual.svg")

# ---- Función auxiliar para heatmaps de betas ----
def heatmap_beta(columna, vcenter, etiqueta_color, archivo):
    matriz = np.full((10, 10), np.nan)
    for _, fila in datos.iterrows():
        i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
        matriz[i, j] = fila[columna]
    fig, ax = plt.subplots(figsize=(9, 7))
    vmax_b = max(abs(np.nanmax(matriz) - vcenter), abs(np.nanmin(matriz) - vcenter))
    norm_b = TwoSlopeNorm(vmin=vcenter - vmax_b, vcenter=vcenter, vmax=vcenter + vmax_b)
    im = ax.imshow(matriz, cmap="RdBu_r", norm=norm_b, aspect="equal")
    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            dist = abs(valor - vcenter)
            color_texto = "white" if dist > vmax_b * 0.55 else "black"
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                    fontsize=8, color=color_texto)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
    ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
    ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
    ax.set_ylabel("Decil Tamaño (1 = pequeña   ←→   10 = grande)")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(etiqueta_color, rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, archivo), format="svg", bbox_inches="tight")
    plt.close()
    print(f"✓ {archivo}")

# ---- Gráfico 2: beta_MKT centrada en 1 ----
heatmap_beta("beta_MKT", 1.0, "Beta de mercado ($\\hat{\\beta}_{i,MKT}$)",
             "02_heatmap_beta_MKT_FF3_anual.svg")

# ---- Gráfico 3: beta_SMB centrada en 0 ----
heatmap_beta("beta_SMB", 0.0, "Beta SMB ($\\hat{\\beta}_{i,SMB}$)",
             "03_heatmap_beta_SMB_FF3_anual.svg")

# ---- Gráfico 4: beta_HML centrada en 0 ----
heatmap_beta("beta_HML", 0.0, "Beta HML ($\\hat{\\beta}_{i,HML}$)",
             "04_heatmap_beta_HML_FF3_anual.svg")

# ---- Gráfico 5: Heatmap alfas FF3 ----
matriz_alfa = np.full((10, 10), np.nan)
for _, fila in datos.iterrows():
    i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
    matriz_alfa[i, j] = fila["alpha"] * 100

fig, ax = plt.subplots(figsize=(9, 7))
vmax = np.nanmax(np.abs(matriz_alfa))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(matriz_alfa, cmap="RdYlGn", norm=norm, aspect="equal")
for i in range(10):
    for j in range(10):
        valor = matriz_alfa[i, j]
        color_texto = "white" if abs(valor) > vmax * 0.6 else "black"
        ax.text(j, i, f"{valor:.1f}", ha="center", va="center",
                fontsize=8, color=color_texto)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = pequeña   ←→   10 = grande)")
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Alfa FF3 anual (%)", rotation=270, labelpad=20)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "05_heatmap_alfas_FF3_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 5: 05_heatmap_alfas_FF3_anual.svg")

# ---- Gráfico 6: t-stats alfas FF3 ----
fig, ax = plt.subplots(figsize=(9, 5.5))
t_stats = datos["alpha_t"]
ax.hist(t_stats, bins=20, color="steelblue", edgecolor="black", alpha=0.75)
ax.axvline(-1.96, color="red", linestyle="--", linewidth=1.5,
           label="Umbrales $\\pm 1.96$ (significación al 5%)")
ax.axvline(1.96, color="red", linestyle="--", linewidth=1.5)
ax.axvline(0, color="black", linewidth=1)
n_sig_5 = (t_stats.abs() > 1.96).sum()
n_sig_1 = (t_stats.abs() > 2.58).sum()
texto = (f"Alfas significativos al 5%: {n_sig_5} / 100\n"
         f"Alfas significativos al 1%: {n_sig_1} / 100\n"
         f"Esperados bajo $H_0$ al 5%: ~5")
ax.text(0.02, 0.97, texto, transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
ax.set_xlabel("t-estadístico de $\\hat{\\alpha}_i$ (FF3)")
ax.set_ylabel("Número de carteras")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "06_tstats_alfas_FF3_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 6: 06_tstats_alfas_FF3_anual.svg")

print(f"\n✓ Los 6 gráficos están en '{CARPETA_GRAFICOS}/'")
