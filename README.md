# Contraste empírico del CAPM y del modelo de tres factores (1963–2026)

> **Trabajo Fin de Grado en Economía** — Universidad de Santiago de Compostela
> **Autor:** Miguel Nicolás Suárez Crespo
> **Director:** Antonio Rodríguez Sampayo
> **Curso académico:** 2025–2026

---

## Resumen

Este repositorio contiene el código y los resultados del contraste empírico del **CAPM** (Sharpe, 1964; Lintner, 1965; Mossin 1966) y del **modelo de tres factores de Fama y French** (FF3, 1993), llevado a cabo como parte del Trabajo Fin de Grado en Economía.

El análisis replica y extiende la metodología en dos etapas de **Fama y MacBeth (1973)** sobre las 100 carteras *Size × Book-to-Market* construidas por Kenneth R. French, ampliando el período muestral original de Fama y French (1992) hasta abril de 2026. Se realizan los contrastes con **datos mensuales** (`T = 754` observaciones) —análisis principal en el TFG— y con **datos anuales** (`T = 62` años) como ejercicio complementario de robustez.

Los principales hallazgos confirman el **rechazo empírico del CAPM** (intercepto significativamente distinto de cero, prima de mercado sin precio detectable en la sección cruzada, falta de linealidad y violación de la no remuneración del riesgo idiosincrático) y muestran que el **modelo de tres factores mejora sustancialmente** el ajuste, aunque sin agotar la sección cruzada de rendimientos.

---

## Estructura del repositorio

```
TFG-Contraste-CAPM-APT-1963-2026/
│
├── datos/                          Datos primarios (CSVs originales de Kenneth R. French)
│   ├── datos_100_carteras.csv
│   └── datos_factores.csv
│
├── scripts/                        Código fuente del análisis
│   ├── capm_mensual/               Contraste CAPM con datos mensuales
│   ├── capm_anual/                 Contraste CAPM con datos anuales (robustez)
│   ├── ff3_mensual/                Contraste FF3 con datos mensuales
│   └── ff3_anual/                  Contraste FF3 con datos anuales (robustez)
│
├── resultados/                     Tablas y figuras generadas por los scripts
│   ├── graficos/                   Figuras en formato vectorial (SVG)
│   │   ├── capm_mensual/
│   │   ├── capm_anual/
│   │   ├── ff3_mensual/
│   │   └── ff3_anual/
│   └── tablas/                     Tablas exportadas a CSV
│       ├── capm_mensual/
│       ├── capm_anual/
│       ├── ff3_mensual/
│       └── ff3_anual/
│
├── README.md                       Este archivo
└── requirements.txt                Dependencias de Python
```

---

## Datos

Los datos provienen de la base pública mantenida por **Kenneth R. French** en el Tuck School of Business (Dartmouth College), accesible en:
[https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html]

Se utilizan dos archivos:

- **Fama/French 3 Factors** — Exceso de rendimiento de la cartera de mercado, tipo libre de riesgo y factores SMB y HML.
- **100 Portfolios Formed on Size and Book-to-Market (10×10)** — 100 carteras *value-weighted* construidas con todas las acciones cotizadas en NYSE, AMEX y NASDAQ.

**Período muestral:** julio 1963 – abril 2026 (`T = 754` observaciones mensuales; `T = 62` años naturales completos para el análisis anual).

---

## Instalación y ejecución

### Requisitos

- Python ≥ 3.10
- Las dependencias están listadas en `requirements.txt`.

### Instalación

```bash
git clone https://github.com/miguelnicosua-bit/TFG-Contraste-CAPM-APT-1963-2026.git
cd TFG-Contraste-CAPM-APT-1963-2026
pip install -r requirements.txt
```

### Orden de ejecución

Los scripts se ejecutan en orden numérico dentro de cada bloque. El **paso 0 de cada bloque genera los archivos intermedios** (`.pkl`) en las carpetas `datos_procesados/` y `datos_procesados_anual/`, que los pasos siguientes consumen. Estas carpetas se crean automáticamente al ejecutar el código.

Los cuatro bloques tienen dependencias entre sí:

- **CAPM mensual:** independiente, se ejecuta primero.
- **FF3 mensual:** reutiliza `datos_procesados/` (requiere haber ejecutado el bloque CAPM mensual).
- **CAPM anual:** reutiliza `datos_procesados/` (requiere haber ejecutado el bloque CAPM mensual).
- **FF3 anual:** requiere los tres bloques anteriores.

**Orden recomendado:** CAPM mensual → FF3 mensual → CAPM anual → FF3 anual.

Desde la raíz del repositorio:

**Bloque 1 — CAPM mensual:**
```bash
python scripts/capm_mensual/paso0_limpieza.py
python scripts/capm_mensual/paso1_betas.py
python scripts/capm_mensual/paso2_seccion_cruzada.py
python scripts/capm_mensual/paso3_contrastes.py
python scripts/capm_mensual/paso4_graficos.py
python scripts/capm_mensual/paso5_comprobaciones.py
```

**Bloque 2 — FF3 mensual:** análogamente, dentro de `scripts/ff3_mensual/`.
**Bloque 3 — CAPM anual:** análogamente, dentro de `scripts/capm_anual/`.
**Bloque 4 — FF3 anual:** análogamente, dentro de `scripts/ff3_anual/`.

---

## Principales resultados

### CAPM mensual (T = 754, ventana móvil de 60 meses)

| Métrica | Valor |
|---|---|
| β media (estática) | 1,082 |
| R² medio | 0,645 |
| \|α̂\| anualizado medio | 2,376% |
| Alfas significativos al 5% (de 100) | 34 |
| Alfas significativos al 1% (de 100) | 18 |
| γ̂₀ anualizada (móvil) | 9,56%*** |
| γ̂₁ anualizada (móvil) | −0,71% (n.s.) |
| Prima realizada del mercado | 7,17% anual |

**Contrastes formales:**

| Contraste | Estadístico | p-valor | Decisión |
|---|---|---|---|
| GRS: H₀: αⱼ = 0 ∀j | F = 2,26 | < 0,001 | Rechaza CAPM |
| Linealidad: H₀: γ₂ = 0 (β²) | t = −3,28 | 0,001 | Rechaza CAPM |
| Riesgo idiosincrático: H₀: γ₃ = 0 | t = −0,14 | 0,89 | Compatible |
| Heterocedasticidad (White, 5%) | 82/100 carteras rechazan | — | Justifica ventana móvil |

El CAPM se rechaza por dos vías formales independientes (GRS y no linealidad) y por la ausencia de una prima detectable en la segunda etapa, manifestación de la **anomalía de la LMA plana** documentada por Black, Jensen y Scholes (1972). Solo el contraste sobre el riesgo idiosincrático resulta compatible con el modelo.

### Modelo de tres factores (FF3) mensual

| Métrica | CAPM | FF3 | Variación |
|---|---|---|---|
| R² medio | 0,645 | 0,800 | +15,5 pp |
| \|α̂\| anualizado medio | 2,376% | 1,467% | −38% |
| Alfas significativos al 5% | 34 | 21 | −38% |
| Alfas significativos al 1% | 18 | 8 | −56% |
| Test GRS | F = 2,26 | F = 2,15 | Sigue rechazando |
| Heterocedasticidad (White, 5%) | 82/100 | 95/100 | Justifica ventana móvil |

**Segunda etapa móvil FF3:**

| Coeficiente | Estimación anual | t | Prima realizada |
|---|---|---|---|
| γ̂₀ | 10,62%*** | 5,90 | — |
| γ̂_MKT | −3,33% | −1,95 (p = 0,051) | 7,17% |
| γ̂_SMB | 1,55% | 1,23 (n.s.) | 1,77% |
| γ̂_HML | 2,91%** | 2,36 | 3,56% |

El factor HML aporta una prima positiva y significativa, próxima a la prima realizada (3,56%): el efecto valor sí tiene precio en la sección cruzada. La pendiente del factor de mercado mantiene el signo negativo del CAPM y queda al borde de la significación, manifestación persistente de la anomalía LMA plana.

### Análisis anual (T = 62, ventana móvil de 10 años, 1964–2025)

Los resultados con datos anuales son **cualitativamente coherentes** con los mensuales:

| | γ̂₀ anual (%) | γ̂₁ / γ̂_MKT anual (%) | γ̂_SMB | γ̂_HML |
|---|---|---|---|---|
| **CAPM anual (móvil)** | 9,85*** | −0,63 (n.s.) | — | — |
| **FF3 anual (móvil)** | 8,31*** | −0,16 (n.s.) | 1,32 (n.s.) | 3,09** |

El test GRS **no es aplicable** con datos anuales (T = 62 < N = 100), limitación estructural del análisis con 100 carteras a baja frecuencia. La validez del rechazo conjunto del CAPM y del FF3 queda apoyada por el análisis mensual. El test de White detecta menos heterocedasticidad (CAPM anual: 4/100; FF3 anual: 37/100), reflejo de que la agregación temporal suaviza la dependencia de la varianza condicional.

---

## Validación del código

Cada bloque incluye un script de comprobaciones que verifica numéricamente la corrección del código:

- `scripts/capm_mensual/paso5_comprobaciones.py`
- `scripts/capm_anual/comprobaciones_capm_anual.py`
- `scripts/ff3_mensual/paso4_ff3_comprobaciones.py`
- `scripts/ff3_anual/comprobaciones_ff3_anual.py`

Cada script ejecuta cinco tests independientes (identidades algebraicas, replicación manual de las regresiones, replicación paso a paso del test GRS, coherencia entre estimadores y consistencia entre frecuencias). Todos los tests devuelven `✓ PASA`.

---

## Referencias principales

Las siguientes referencias respaldan tanto el contenido teórico y empírico del TFG como el código y los desarrollos recogidos en el notebook del apéndice computacional.

- Black, F., Jensen, M. C. y Scholes, M. (1972). The Capital Asset Pricing Model: Some empirical tests. En M. C. Jensen (Ed.), *Studies in the theory of capital markets* (pp. 79–121). Praeger.

- Campbell, J. Y., Lo, A. W. y MacKinlay, A. C. (1997). *The econometrics of financial markets*. Princeton University Press.

- Danthine, J.-P. y Donaldson, J. B. (2014). *Intermediate financial theory* (3.ª ed.). Academic Press.

- Fama, E. F. y French, K. R. (1992). The cross-section of expected stock returns. *The Journal of Finance*, 47(2), 427–465. https://doi.org/10.1111/j.1540-6261.1992.tb04398.x

- Fama, E. F. y French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56. https://doi.org/10.1016/0304-405X(93)90023-5

- Fama, E. F. y French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22. https://doi.org/10.1016/j.jfineco.2014.10.010

- Gibbons, M. R., Ross, S. A. y Shanken, J. (1989). A test of the efficiency of a given portfolio. *Econometrica*, 57(5), 1121–1152. https://doi.org/10.2307/1913625

- Jensen, M. C. (1968). The performance of mutual funds in the period 1945–1964. *The Journal of Finance*, 23(2), 389–416. https://doi.org/10.1111/j.1540-6261.1968.tb00815.x

- Lintner, J. (1965). The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets. *The Review of Economics and Statistics*, 47(1), 13–37. https://doi.org/10.2307/1924119

- Marín, J. M. y Rubio, G. (2001). *Economía financiera* (1.ª ed.). Antoni Bosch.

- Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77–91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x

- Mossin, J. (1966). Equilibrium in a capital asset market. *Econometrica*, 34(4), 768–783. https://doi.org/10.2307/1910098

- Rodríguez Sampayo, A. (curso académico 2025–2026). *Apuntes de Microeconomía de los Mercados Financieros*. Material docente no publicado, Universidad de Santiago de Compostela.

- Ross, S. A. (1976). The arbitrage theory of capital asset pricing. *Journal of Economic Theory*, 13(3), 341–360. https://doi.org/10.1016/0022-0531(76)90046-6

- Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. *The Journal of Finance*, 19(3), 425–442. https://doi.org/10.1111/j.1540-6261.1964.tb02865.x

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838. https://doi.org/10.2307/1912934

---

## Cómo citar este trabajo

> Suárez Crespo, M. N. (2026). *Contraste empírico del CAPM y del modelo de tres factores de Fama y French (1963–2026)*. Trabajo Fin de Grado, Facultad de Ciencias Económicas y Empresariales, Universidad de Santiago de Compostela.

---

## Asistencia de IA

El desarrollo de este repositorio ha contado con la asistencia del modelo de inteligencia artificial **Claude (Anthropic)** en tareas concretas:

- Generación inicial y depuración de los scripts en Python para el análisis empírico (regresiones, test GRS, generación de figuras).
- Redacción y revisión estilística de este README.
- El código en Python contenido en este Jupyter Notebook.

Todo el código ha sido posteriormente revisado, ejecutado y validado por el autor. Las decisiones metodológicas, la redacción e interpretación de los resultados, la elección de los contrastes y la estructura general del análisis son responsabilidad exclusiva del autor.

---

## Licencia

Este repositorio se publica con fines exclusivamente académicos. Los datos pertenecen a sus respectivos titulares (Kenneth R. French, CRSP, COMPUSTAT).
