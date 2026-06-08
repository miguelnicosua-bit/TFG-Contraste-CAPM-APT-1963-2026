"""
PASO 3 FF3 ANUAL - Gráficos del modelo de tres factores con datos anuales
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Cinco figuras en alta resolución (300 dpi):
  1. Heatmap 10x10 de alfas FF3 (anuales).
  2. Comparación lado a lado: alfas CAPM anual vs alfas FF3 anual.
  3. Heatmap 10x10 de las betas a SMB (anual).
  4. Heatmap 10x10 de las betas a HML (anual).
  5. Comparación FF3 mensual vs FF3 anual (heatmaps de alfas).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Configuración para gráficos vectoriales con texto seleccionable
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.colors import TwoSlopeNorm

CARPETA_ANUAL = "datos_procesados_anual"
CARPETA_GRAFICOS = "resultados/graficos/ff3_anual"
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

primera_ff3_a   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_ff3_anual.pkl"))
primera_capm_a  = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_anual.pkl"))
primera_ff3_m   = pd.read_pickle(os.path.join("datos_procesados", "primera_etapa_ff3.pkl"))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 100,
})

# ========================================================================
# FUNCIÓN AUXILIAR
# ========================================================================
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

parsed = pd.DataFrame(
    [parsear_nombre(c) for c in primera_ff3_a.index],
    columns=["size", "bm"], index=primera_ff3_a.index
)
ff3_a  = pd.concat([primera_ff3_a, parsed], axis=1)
capm_a = pd.concat([primera_capm_a, parsed], axis=1)
ff3_m  = pd.concat([primera_ff3_m, parsed], axis=1)

def matriz_10x10(df, columna, factor=1.0):
    M = np.full((10, 10), np.nan)
    for _, fila in df.iterrows():
        i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
        M[i, j] = fila[columna] * factor
    return M

# ========================================================================
# GRÁFICO 1: HEATMAP DE ALFAS FF3 (anuales)
# ========================================================================
matriz_alfa_ff3 = matriz_10x10(ff3_a, "alpha", 100)  # en %

fig, ax = plt.subplots(figsize=(9, 7))
# Misma escala que el CAPM anual (para comparación visual)
vmax_capm_a = np.nanmax(np.abs(matriz_10x10(capm_a, "alpha", 100)))
norm = TwoSlopeNorm(vmin=-vmax_capm_a, vcenter=0, vmax=vmax_capm_a)
im = ax.imshow(matriz_alfa_ff3, cmap="RdYlGn", norm=norm, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_alfa_ff3[i, j]
        color_texto = "white" if abs(valor) > vmax_capm_a * 0.6 else "black"
        ax.text(j, i, f"{valor:.1f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = small   ←→   10 = big)")
ax.set_title("Alfas anuales (%) bajo el modelo de tres factores\n"
             "(Escala fija para comparar con el CAPM anual)")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Alfa FF3 anual (%)", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "06_heatmap_alfas_ff3_anual.svg"),
             format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 1 guardado: 06_heatmap_alfas_ff3_anual.svg")

# ========================================================================
# GRÁFICO 2: COMPARACIÓN CAPM ANUAL vs FF3 ANUAL
# ========================================================================
matriz_alfa_capm = matriz_10x10(capm_a, "alpha", 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

vmax_global = max(np.nanmax(np.abs(matriz_alfa_capm)),
                  np.nanmax(np.abs(matriz_alfa_ff3)))
norm_g = TwoSlopeNorm(vmin=-vmax_global, vcenter=0, vmax=vmax_global)

im1 = ax1.imshow(matriz_alfa_capm, cmap="RdYlGn", norm=norm_g, aspect="equal")
for i in range(10):
    for j in range(10):
        v = matriz_alfa_capm[i, j]
        c = "white" if abs(v) > vmax_global * 0.6 else "black"
        ax1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=c)
ax1.set_xticks(range(10)); ax1.set_yticks(range(10))
ax1.set_xticklabels([f"BM{j+1}" for j in range(10)], fontsize=9)
ax1.set_yticklabels([f"ME{i+1}" for i in range(10)], fontsize=9)
ax1.set_xlabel("Decil BM (growth ←→ value)")
ax1.set_ylabel("Decil Tamaño (small ←→ big)")
ax1.set_title("(a) Alfas bajo el CAPM (anual)")

im2 = ax2.imshow(matriz_alfa_ff3, cmap="RdYlGn", norm=norm_g, aspect="equal")
for i in range(10):
    for j in range(10):
        v = matriz_alfa_ff3[i, j]
        c = "white" if abs(v) > vmax_global * 0.6 else "black"
        ax2.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=c)
ax2.set_xticks(range(10)); ax2.set_yticks(range(10))
ax2.set_xticklabels([f"BM{j+1}" for j in range(10)], fontsize=9)
ax2.set_yticklabels([f"ME{i+1}" for i in range(10)], fontsize=9)
ax2.set_xlabel("Decil BM (growth ←→ value)")
ax2.set_title("(b) Alfas bajo el modelo de tres factores (anual)")

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label("Alfa anual (%)", rotation=270, labelpad=20)

fig.suptitle("Comparación del ajuste con datos anuales: alfas bajo CAPM vs FF3",
             fontsize=13, y=1.00)
plt.savefig(os.path.join(CARPETA_GRAFICOS, "07_comparacion_ff3_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 2 guardado: 07_comparacion_ff3_anual.svg")

# ========================================================================
# GRÁFICO 3: HEATMAP BETAS SMB (anuales)
# ========================================================================
matriz_smb = matriz_10x10(ff3_a, "beta_SMB")

fig, ax = plt.subplots(figsize=(9, 7))
vmax_smb = max(abs(np.nanmax(matriz_smb)), abs(np.nanmin(matriz_smb)))
norm_smb = TwoSlopeNorm(vmin=-vmax_smb, vcenter=0, vmax=vmax_smb)
im = ax.imshow(matriz_smb, cmap="RdBu_r", norm=norm_smb, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_smb[i, j]
        color_texto = "white" if abs(valor) > vmax_smb * 0.6 else "black"
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = small   ←→   10 = big)")
ax.set_title("Sensibilidad al factor SMB ($\\hat{\\beta}^{SMB}_j$) — Datos anuales")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("$\\hat{\\beta}^{SMB}$", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "08_heatmap_betas_SMB_ff3_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 3 guardado: 08_heatmap_betas_SMB_ff3_anual.svg")

# ========================================================================
# GRÁFICO 4: HEATMAP BETAS HML (anuales)
# ========================================================================
matriz_hml = matriz_10x10(ff3_a, "beta_HML")

fig, ax = plt.subplots(figsize=(9, 7))
vmax_hml = max(abs(np.nanmax(matriz_hml)), abs(np.nanmin(matriz_hml)))
norm_hml = TwoSlopeNorm(vmin=-vmax_hml, vcenter=0, vmax=vmax_hml)
im = ax.imshow(matriz_hml, cmap="RdBu_r", norm=norm_hml, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_hml[i, j]
        color_texto = "white" if abs(valor) > vmax_hml * 0.6 else "black"
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = small   ←→   10 = big)")
ax.set_title("Sensibilidad al factor HML ($\\hat{\\beta}^{HML}_j$) — Datos anuales")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("$\\hat{\\beta}^{HML}$", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "09_heatmap_betas_HML_ff3_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 4 guardado: 09_heatmap_betas_HML_ff3_anual.svg")

# ========================================================================
# GRÁFICO 5: COMPARACIÓN FF3 MENSUAL vs FF3 ANUAL (heatmaps alfas)
# ========================================================================
matriz_ff3_m = matriz_10x10(ff3_m, "alpha", 12 * 100)  # mensual a anual
matriz_ff3_a = matriz_10x10(ff3_a, "alpha", 100)       # ya en términos anuales

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

vmax_global = max(np.nanmax(np.abs(matriz_ff3_m)),
                  np.nanmax(np.abs(matriz_ff3_a)))
norm_g = TwoSlopeNorm(vmin=-vmax_global, vcenter=0, vmax=vmax_global)

im1 = ax1.imshow(matriz_ff3_m, cmap="RdYlGn", norm=norm_g, aspect="equal")
for i in range(10):
    for j in range(10):
        v = matriz_ff3_m[i, j]
        c = "white" if abs(v) > vmax_global * 0.6 else "black"
        ax1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=c)
ax1.set_xticks(range(10)); ax1.set_yticks(range(10))
ax1.set_xticklabels([f"BM{j+1}" for j in range(10)], fontsize=9)
ax1.set_yticklabels([f"ME{i+1}" for i in range(10)], fontsize=9)
ax1.set_xlabel("Decil BM (growth ←→ value)")
ax1.set_ylabel("Decil Tamaño (small ←→ big)")
ax1.set_title("(a) Alfas FF3 mensuales (anualizados)")

im2 = ax2.imshow(matriz_ff3_a, cmap="RdYlGn", norm=norm_g, aspect="equal")
for i in range(10):
    for j in range(10):
        v = matriz_ff3_a[i, j]
        c = "white" if abs(v) > vmax_global * 0.6 else "black"
        ax2.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=c)
ax2.set_xticks(range(10)); ax2.set_yticks(range(10))
ax2.set_xticklabels([f"BM{j+1}" for j in range(10)], fontsize=9)
ax2.set_yticklabels([f"ME{i+1}" for i in range(10)], fontsize=9)
ax2.set_xlabel("Decil BM (growth ←→ value)")
ax2.set_title("(b) Alfas FF3 anuales")

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label("Alfa anual (%)", rotation=270, labelpad=20)

fig.suptitle("Robustez del modelo FF3 a la frecuencia: alfas mensuales vs anuales",
             fontsize=13, y=1.00)
plt.savefig(os.path.join(CARPETA_GRAFICOS, "10_comparacion_ff3_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 5 guardado: 10_comparacion_ff3_anual.svg")

print(f"\n✓ Los 5 gráficos del FF3 anual están en '{CARPETA_GRAFICOS}/'")
print("✓ Resolución 300 dpi, listos para insertar en el GitHub")
