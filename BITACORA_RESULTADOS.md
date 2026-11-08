# Bitácora de Resultados y Verificación — Taller 3 (OBS)

**Asignatura:** Inteligencia Computacional II  
**Tema:** Diseño de redes neuronales mediante algoritmos destructivos — *Optimal Brain Surgeon (OBS)*  

---

## Organización de Carpetas del Proyecto
- `01_Documentacion/`: `TallerIII.pdf`, `OBS_guia_de_estudio.pdf`, `647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon.pdf`.
- `02_Datos/`: `datos_taller3.mat` (y copia en raíz para compatibilidad rápida).
- `03_Scripts/`: Scripts de MATLAB de cada punto (`analisis_exploratorio_taller3.m`, etc.).
- `04_Figuras/`: Figuras generadas en alta resolución (`.png`).
- `05_Resultados_Parciales/`: Modelos entrenados, matrices Hessianas y resultados de experimentos multi-semilla.
- `06_Entregables/`: Archivos finales `.mat` (`OBS_codigo.MAT` y `pesos_codigo.mat`).

---

## Punto 1: Análisis Exploratorio de la Base de Datos (`datos_taller3.mat`)

### 1. Ecuaciones y Métodos Aplicados
- **Índice Discriminante de Fisher ($J_i$):** Mide la separabilidad lineal univariada entre la Clase 0 y la Clase 1 para cada característica $i \in \{1, \dots, 10\}$:
  $$J_i = \frac{(\mu_{i,0} - \mu_{i,1})^2}{\sigma_{i,0}^2 + \sigma_{i,1}^2}$$
- **Matriz de Correlación de Pearson ($\mathbf{R}$) y Descomposición Espectral (PCA):** Evalúa la colinealidad entre variables de entrada mediante los autovalores $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{10}$ de $\mathbf{R}$ y su número de condición $\kappa(\mathbf{R}) = \frac{\lambda_{\max}}{\lambda_{\min}}$.
- **Estadístico de Kolmogorov-Smirnov de dos muestras ($KS$):** Cuantifica la distancia máxima entre las funciones de distribución acumulada empíricas de entrenamiento y prueba:
  $$KS_i = \sup_x |F_{\text{train}, i}(x) - F_{\text{test}, i}(x)|$$

### 2. Resultados Numéricos Verificados
- **Dimensiones e Integridad:**
  - `x_train`: $1600 \times 10$ (`double`), $0$ valores `NaN` o `Inf`.
  - `y_train`: $1600 \times 1$ (`double`), clases perfectamente balanceadas: **800 muestras de Clase 0 (50.0%)** y **800 muestras de Clase 1 (50.0%)**.
  - `x_test`: $13\,888 \times 10$ (`double`), $0$ valores `NaN` o `Inf`.
- **Relación Capacidad de Red (`10-60-1`) vs. Muestras ($P$ vs. $n$):**
  - Parámetros totales con biases: $n = 10(60) + 60 + 60(1) + 1 = 721$ pesos ($660$ pesos sinápticos sin biases).
  - Razón global $P/n = 1600 / 721 = 2.22 > 1$, cumpliendo la condición de rango para la matriz Hessiana empírica.
- **Hallazgos Estadísticos Clave:**
  1. **Disparidad de escalas:** $x_4 \in [0.021, 0.877]$ y $x_5 \in [0.005, 0.637]$ frente a $x_6 \in [-449.95, 473.07]$ y $x_{10} \in [5.70, 430.12]$. Hace obligatoria la normalización *z-score* antes de entrenar y calcular derivadas de segundo orden.
  2. **Estructura no lineal (Núcleo vs. Halo):** Variables como $x_6, x_7, x_8$ presentan medias similares entre clases pero desviaciones estándar en la Clase 0 que duplican a las de la Clase 1 ($\sigma_{C0} = 84.61$ vs. $\sigma_{C1} = 38.94$ en $x_6$). La variable con mayor discriminabilidad individual es $x_9$ ($J_9 = 0.5343$, $\text{AUC} = 0.7797$), seguida de $x_1$ ($J_1 = 0.2055$) y $x_2$ ($J_2 = 0.1363$). Variables como $x_8$ ($J_8 = 0.0006$), $x_5$ ($J_5 = 0.0005$) y $x_4$ ($J_4 = 0.0013$) tienen aporte lineal casi nulo.
  3. **Colinealidad severa ($\implies$ Hessiana no diagonal):** Alta correlación entre $\{x_1, x_2, x_3, x_4, x_5\}$, destacando $r(x_4, x_5) = 0.976$ y $r(x_3, x_4) = -0.850$. El menor autovalor de correlación es $\lambda_{10} = 0.0201$ ($\kappa(\mathbf{R}) = 210.2$), lo que garantiza fuertes términos cruzados fuera de la diagonal en $\mathbf{H}$ y justifica el uso de OBS sobre OBD o Magnitud.
  4. **Alineación `Train` vs. `Test`:** Todos los estadísticos $KS \in [0.0160, 0.0405]$, confirmando que `x_train` y `x_test` provienen exactamente de la misma distribución poblacional.

### 3. Rutas de Scripts y Figuras Generadas
- **Script:** `03_Scripts/analisis_exploratorio_taller3.m`
- **Figuras:**
  - `04_Figuras/fig1_escalas_y_dispersion.png`: Boxcharts de rango original y comparación de desviaciones estándar por clase ($\sigma_{C0} > \sigma_{C1}$).
  - `04_Figuras/fig2_distribuciones_clases.png`: Histogramas de densidad condicionales por clase para las 10 variables junto con su índice de Fisher $J$.
  - `04_Figuras/fig3_correlacion_y_pca.png`: Mapa de calor de la matriz de correlación $10 \times 10$ y espectro de varianza explicada acumulada (PCA).
  - `04_Figuras/fig4_geometria_y_traintest.png`: Diagrama de dispersión no lineal ($x_9$ vs $x_6$, núcleo vs halo) y curvas CDF empíricas superpuestas entre entrenamiento y prueba.

---

## Punto 2: Partición Estratificada, Estandarización y Benchmark Multi-Algoritmo de la Red Base (MLP 10-60-1)

### 1. Ecuaciones y Métodos Aplicados
- **Empaquetamiento Lexicográfico ($\mathbf{w} \in \mathbb{R}^{721 \times 1}$):**
  $$\mathbf{w} = [\operatorname{vec}(\mathbf{W}_1)^T, \mathbf{b}_1^T, \operatorname{vec}(\mathbf{W}_2)^T, b_2]^T, \quad \mathbf{W}_1 \in \mathbb{R}^{60 \times 10},\; \mathbf{b}_1 \in \mathbb{R}^{60 \times 1},\; \mathbf{W}_2 \in \mathbb{R}^{1 \times 60},\; b_2 \in \mathbb{R}$$
- **Función de Costo MSE + Regularización $L_2$ / Evidencia Bayesiana de MacKay (Hassibi & Stork, Ec. 7; MacKay, 1992):**
  $$E(\mathbf{w}) = \frac{1}{2P}\sum_{k=1}^{P}\left(t^{[k]} - o^{[k]}\right)^2, \qquad J(\mathbf{w}) = E(\mathbf{w}) + \frac{\lambda_{\text{eff}}}{2}\|\mathbf{w}\|_2^2, \qquad \gamma_{\text{eff}} = \sum_{i=1}^{n}\frac{\mu_i}{\mu_i + \lambda_{\text{eff}}}$$
- **Benchmark de 6 Familias de Algoritmos de Entrenamiento ($\times$ 3 Configuraciones de Activación = 156 Redes):**
  1. **`trainbr` (Regularización Bayesiana de MacKay / Foresee & Hagan, 1997):** Actualización empírica bayesiana de $\lambda_{\text{eff}} = \frac{\alpha}{P\beta} \propto \frac{\gamma_{\text{eff}}}{P-\gamma_{\text{eff}}}\frac{2 E_D}{\|\mathbf{w}\|_2^2}$ usando el espectro de la Hessiana de Gauss-Newton $\mathbf{H}_{\text{GN}} = \frac{1}{P}\mathbf{J}\mathbf{J}^T$.
  2. **`trainbfg` (Cuasi-Newton BFGS / L-BFGS):** Recursión secante de dos bucles ($m=25$) con búsqueda de línea de Armijo.
  3. **`trainscg` (Gradiente Conjugado Escalado de Møller, 1993):** Región de confianza adaptativa de segundo orden sobre $\mathbf{s}_k = \mathbf{H}\mathbf{p}_k$.
  4. **`traincgp` (Gradiente Conjugado de Polak-Ribière):** Direcciones conjugadas con reinicio automático $\beta_k^{\text{PR}+}$.
  5. **`trainrp` (Resilient Backpropagation de Riedmiller + Pulido Local):** Paso adaptativo por signo de derivada parcial ($\eta^+ = 1.2, \eta^- = 0.5$) con pulido final de curvatura.
  6. **`trainlm` (Levenberg-Marquardt):** Paso Gauss-Newton amortiguado $(\mathbf{H}_{\text{GN}} + \mu\mathbf{I})^{-1}\mathbf{g}$.

### 2. Resultados Numéricos Verificados
- **Partición Estratificada 70% / 15% / 15% (`rng(42)`):**
  - Entrenamiento: $P_{\text{tr}} = 1120$ ($560$ Clase 0 / $560$ Clase 1), cumpliendo $P_{\text{tr}} > n = 721$ ($P_{\text{tr}}/n = 1.5534$).
  - Validación: $P_{\text{val}} = 240$ ($120$ Clase 0 / $120$ Clase 1).
  - Prueba Interna (Hold-out): $P_{\text{te\_int}} = 240$ ($120$ Clase 0 / $120$ Clase 1).
- **Verificación de Gradiente Analítico vs. Diferencias Finitas ($\epsilon = 10^{-6}$):**
  - Error relativo máximo del gradiente: $2.92 \times 10^{-11} < 10^{-8}$.
- **Comparativa Global por Familia de Entrenamiento (156 Redes):**
  - `trainrp`: $\text{Err}_{\text{val}} = 19.41 \pm 1.29\%$ (Mín $17.08\%$), $\|\mathbf{w}\|_2 = 8.61$, $\gamma_{\text{eff}} = 134.5$.
  - `trainscg`: $\text{Err}_{\text{val}} = 19.44 \pm 0.85\%$ (Mín $17.50\%$), $\|\mathbf{w}\|_2 = 8.79$, $\gamma_{\text{eff}} = 139.7$.
  - `trainbr`: $\text{Err}_{\text{val}} = 19.50 \pm 1.21\%$ (Mín $17.08\%$, $\text{Err}_{\text{te\_int}}$ Mín $13.33\%$), $\|\mathbf{w}\|_2 = 11.53$, $\gamma_{\text{eff}} = 137.1$.
  - `traincgp`: $\text{Err}_{\text{val}} = 19.72 \pm 1.05\%$ (Mín $17.92\%$), $\|\mathbf{w}\|_2 = 7.02$, $\gamma_{\text{eff}} = 108.2$.
  - `trainbfg`: $\text{Err}_{\text{val}} = 19.76 \pm 1.46\%$ (Mín $16.67\%$), $\|\mathbf{w}\|_2 = 18.28$, $\gamma_{\text{eff}} = 152.5$.
  - `trainlm`: $\text{Err}_{\text{val}} = 20.71 \pm 2.18\%$ (Mín $17.08\%$), $\|\mathbf{w}\|_2 = 25.18$, $\gamma_{\text{eff}} = 236.0$ (presenta sobreajuste por inflación de $\|\mathbf{w}\|_2$).
- **Las 3 Redes Finalistas Seleccionadas (`top3_modelos` = 2 Mejores No-LM + 1 Mejor LM):**
  1. **Red #1 (Mejor No-LM #1 / Campeona Global — `trainbr`):** `Config B (tanh + lineal)`, $\lambda_{\text{eff}} = 1.0 \times 10^{-3}$, Semilla `2102`.
     - $E_{\text{tr}} = 0.0570$ ($\text{Err}_{\text{tr}} = 15.09\%$, $\text{AUC}_{\text{tr}} = 0.9239$).
     - $E_{\text{val}} = 0.0648$ ($\text{Err}_{\text{val}} = \mathbf{17.08\%}$, $\text{AUC}_{\text{val}} = 0.9040$).
     - $E_{\text{te\_int}} = 0.0572$ ($\text{Err}_{\text{te\_int}} = \mathbf{14.58\%}$, Exactitud $= \mathbf{85.42\%}$, $\text{AUC}_{\text{te\_int}} = 0.9254$).
     - Geometría OBS: $\|\mathbf{w}\|_2 = \mathbf{4.04}$, $\gamma_{\text{eff}} = 126.3$, $556/721$ pesos con $|w_q| < 0.10$, $\|\nabla J\|_2 = 5.56 \times 10^{-3}$, $\|\nabla E\|_2 = 6.72 \times 10^{-3}$.
  2. **Red #2 (Mejor No-LM #2 — `trainrp`):** `Config B (tanh + lineal)`, $\lambda = 5.0 \times 10^{-4}$, Semilla `2504`.
     - $E_{\text{tr}} = 0.0565$ ($\text{Err}_{\text{tr}} = 15.27\%$), $E_{\text{val}} = 0.0636$ ($\text{Err}_{\text{val}} = \mathbf{17.08\%}$), $\text{Err}_{\text{te\_int}} = \mathbf{15.00\%}$ ($\text{AUC}_{\text{te\_int}} = 0.9221$), $\|\mathbf{w}\|_2 = \mathbf{4.86}$, $\gamma_{\text{eff}} = 172.0$, $\|\nabla J\|_2 = 8.8 \times 10^{-3}$.
  3. **Red #3 (Mejor LM — `trainlm`):** `Config C (sigmoide + sigmoide)`, $\lambda = 1.0 \times 10^{-4}$, Semilla `3603`.
     - $E_{\text{tr}} = 0.0550$ ($\text{Err}_{\text{tr}} = 14.02\%$), $E_{\text{val}} = \mathbf{0.0612}$ ($\text{Err}_{\text{val}} = 17.92\%$), $\text{Err}_{\text{te\_int}} = 16.25\%$ ($\text{AUC}_{\text{te\_int}} = 0.9269$), $\|\mathbf{w}\|_2 = 13.68$, $\gamma_{\text{eff}} = 70.4$, $\|\nabla J\|_2 = 9.1 \times 10^{-4}$.

### 3. Rutas de Scripts, Datos y Figuras Generadas
- **Script:** `03_Scripts/paso2_entrenar_red_base.m`
- **Datos y Modelos:** `02_Datos/particion_datos.mat`, `05_Resultados_Parciales/red_base_10_60_1.mat`
- **Figuras:**
  - `04_Figuras/fig5_desempeno_multistart_red_base.png`: Comparativa de las 6 familias de algoritmos en validación, geometría OBS ($\|\mathbf{w}\|_2$ y $\gamma_{\text{eff}}$), mejor red por familia y dispersión resaltando las 3 redes seleccionadas.
  - `04_Figuras/fig6_diagnostico_mejor_red_base.png`: Convergencia de MSE y $\|\nabla J\|_2$, histograma de los 721 pesos iniciales, curvas ROC y matriz de confusión de la Red Campeona (`trainbr`).

---

## Punto 3: Implementación del Motor Matemático de $\mathbf{H}^{-1}$ por Productos Externos (Sobre las 3 Redes Seleccionadas)

### 1. Ecuaciones y Métodos Aplicados
- **Vector de Sensibilidades de la Salida respecto a los Pesos ($\mathbf{X}^{[k]}$, Ecs. 11 y 19–21 de Hassibi & Stork):**
  $$\mathbf{X}^{[k]} = \frac{\partial F(\mathbf{w}, \mathbf{in}^{[k]})}{\partial \mathbf{w}} = \begin{bmatrix} \operatorname{vec}\!\left(f_{\text{out}}'(z_2^{[k]})\, \mathbf{v} \odot f_{\text{hid}}'(\mathbf{z}_1^{[k]})\, (\mathbf{x}^{[k]})^T\right)_{1:600} \\ \left(f_{\text{out}}'(z_2^{[k]})\, \mathbf{v} \odot f_{\text{hid}}'(\mathbf{z}_1^{[k]})\right)_{601:660} \\ \left(f_{\text{out}}'(z_2^{[k]})\, \mathbf{a}_1^{[k]}\right)_{661:720} \\ \left(f_{\text{out}}'(z_2^{[k]})\right)_{721} \end{bmatrix} \in \mathbb{R}^{721 \times 1}$$
- **Matriz Hessiana por Productos Externos (Ec. 12) y Recursión de Sherman-Morrison-Woodbury (Ec. 17):**
  $$\mathbf{H} = \frac{1}{P}\sum_{k=1}^{P}\mathbf{X}^{[k]}(\mathbf{X}^{[k]})^T, \qquad \mathbf{H}_{m+1}^{-1} = \mathbf{H}_m^{-1} - \frac{\mathbf{H}_m^{-1}\mathbf{X}^{[m+1]}(\mathbf{X}^{[m+1]})^T\mathbf{H}_m^{-1}}{P + (\mathbf{X}^{[m+1]})^T\mathbf{H}_m^{-1}\mathbf{X}^{[m+1]}}, \quad \mathbf{H}_0^{-1} = \alpha^{-1}\mathbf{I}_{721}$$
- **Complemento de Schur y Saliencias (Sección 5.2 de la Guía):**
  $$\frac{1}{[\mathbf{H}^{-1}]_{qq}} = H_{qq} + \alpha - \mathbf{h}_q^T(\mathbf{H}_{\text{resto}} + \alpha\mathbf{I})^{-1}\mathbf{h}_q \le H_{qq} + \alpha, \qquad L_q^{\text{OBS}} = \frac{w_q^2}{2[\mathbf{H}^{-1}]_{qq}}$$

### 2. Resultados Numéricos Verificados
- **Verificación de $\mathbf{X}^{[k]}$ contra Diferencias Finitas Centradas ($\epsilon = 10^{-6}$) en las 3 Redes Finalistas:**
  - Red #1 (`trainbr`, `Config B: tanh+lin`): Error rel. máx $= 1.58 \times 10^{-10} < 10^{-8}$.
  - Red #2 (`trainrp`, `Config B: tanh+lin`): Error rel. máx $= 2.16 \times 10^{-10} < 10^{-8}$.
  - Red #3 (`trainlm`, `Config C: sig+sig`): Error rel. máx $= 2.18 \times 10^{-9} < 10^{-8}$.
  - Ref. `Config A (tanh+sig)`: Error rel. máx $= 4.35 \times 10^{-9} < 10^{-8}$.
- **Comparativa Estructural de $\mathbf{H}$ y $\mathbf{H}^{-1}$ (Woodbury, $\alpha = 10^{-6}$) en las 3 Redes Finalistas (`comparativa_3redes_H`):**
  - **Red #1 (`trainbr`, `tanh+lin`):** $\operatorname{Tr}(\mathbf{H}) = 32.292$, $\lambda_{\max} = 9.933$, Rango efectivo $= 449 / 721$, $\kappa(\mathbf{H}+\alpha\mathbf{I}) = 9.93 \times 10^6$, Masa Off-Diag Frobenius $= \mathbf{97.99\%}$ ($99.02\%$ en $L_1$), Sobreestimación OBD/OBS $= \mathbf{3\,964.7\times}$, Error Relativo Woodbury $= \mathbf{1.88 \times 10^{-11}}$.
  - **Red #2 (`trainrp`, `tanh+lin`):** $\operatorname{Tr}(\mathbf{H}) = 36.460$, $\lambda_{\max} = 11.830$, Rango efectivo $= 698 / 721$, $\kappa(\mathbf{H}+\alpha\mathbf{I}) = 1.18 \times 10^7$, Masa Off-Diag Frobenius $= \mathbf{98.72\%}$ ($99.32\%$ en $L_1$), Sobreestimación OBD/OBS $= \mathbf{8\,818.9\times}$, Error Relativo Woodbury $= \mathbf{4.84 \times 10^{-11}}$.
  - **Red #3 (`trainlm`, `sig+sig`):** $\operatorname{Tr}(\mathbf{H}) = 0.723$, $\lambda_{\max} = 0.435$, Rango efectivo $= 320 / 721$, $\kappa(\mathbf{H}+\alpha\mathbf{I}) = 4.35 \times 10^5$, Masa Off-Diag Frobenius $= \mathbf{98.96\%}$ ($99.05\%$ en $L_1$), Sobreestimación OBD/OBS $= \mathbf{537.3\times}$, Error Relativo Woodbury $= \mathbf{4.41 \times 10^{-12}}$.
- **Complemento de Schur y Teorema de Poda en Un Paso (Red #1 `trainbr`):**
  - En los $171$ pesos activos ($|w_q| \ge 0.08$), la correlación de rangos de Spearman cae a $\rho(\text{Mag}, \text{OBS}) = 0.6104$ y $\rho(\text{OBD}, \text{OBS}) = 0.7510$.
  - En el experimento de poda de 1 paso sobre pesos activos ($|w_q| \ge 0.04$), la cirugía de OBS reduce el incremento real del error $|\Delta E_{\text{real}}|$ en **$1\,150.3\times$** frente a OBD ($6.24\times 10^{-10}$ vs $7.17\times 10^{-7}$) y en **$111.8\times$** para $|w_q| \ge 0.08$.

### 3. Rutas de Scripts, Datos y Figuras Generadas
- **Script:** `03_Scripts/paso3_hessiana_obs.m`
- **Datos:** `05_Resultados_Parciales/hessiana_obs_punto3.mat`
- **Figuras:**
  - `04_Figuras/fig7_mapa_hessiana_y_woodbury.png`: Mapas de calor de $\log_{10}|\mathbf{H}|$ y $\log_{10}|\mathbf{H}^{-1}|$ de `Red 1 (trainbr)`, evolución de la estabilidad numérica de Woodbury y espectro de autovalores de $\mathbf{H}$ comparando las 3 redes finalistas.
  - `04_Figuras/fig8_comparativa_saliencias_obs_obd_mag.png`: Reducción de curvatura por complemento de Schur, perfil ordenado de saliencias de las 3 redes finalistas y OBS vs OBD vs Magnitud, reordenamiento de prioridad y verificación empírica $\Delta E(\text{Mag}) \ge \Delta E(\text{OBD}) \gg \Delta E(\text{OBS})$.

---

## Punto 4: Algoritmo de Poda Optimal Brain Surgeon (OBS), Comparativa de las 3 Redes Finalistas y Selección de la Red Óptima

### 1. Ecuaciones y Métodos Aplicados
- **Saliencia y Ajuste Quirúrgico de Segundo Orden (Hassibi & Stork, Ec. 5):**
  $$q^* = \arg\min_{q \in \mathcal{A}} \frac{w_q^2}{2[\mathbf{H}^{-1}]_{qq}}, \qquad \mathbf{w} \leftarrow \mathbf{w} - \frac{w_{q^*}}{[\mathbf{H}^{-1}]_{q^*q^*}}\mathbf{H}^{-1}\mathbf{e}_{q^*}$$
- **Proyección de Gauss-Jordan / Schur de $\mathbf{H}^{-1}$ al Subespacio Reducido (Sección 15.3 Guía):**
  $$\mathbf{H}_{\text{nuevo}}^{-1} = \mathbf{H}^{-1} - \frac{(\mathbf{H}^{-1}\mathbf{e}_{q^*})(\mathbf{H}^{-1}\mathbf{e}_{q^*})^T}{[\mathbf{H}^{-1}]_{q^*q^*}}$$
  combinada con recálculo periódico de $\mathbf{H}^{-1}$ sobre el soporte activo `act_idx = find(mask_act)` y pulido en el subespacio activo con máscara fija en el régimen estructural ($\le 180$ pesos activos).
- **Triple Candado Numérico (Restricción del Profesor):**
  1. `L(~(mask_act & mask_podables)) = inf;`
  2. `mask_act(q_star) = false; w(q_star) = 0;`
  3. `w = w .* double(mask_act);`

### 2. Resultados Numéricos Verificados
- **Verificación del Triple Candado:** En todas las trayectorias completas de $655$ pasos cada una, el $100\%$ de los pesos eliminados quedó en `0.0000000000000000` exacto y `length(unique(historial_q)) == 655` (cero repeticiones de índices).
- **Sensibilidad de OBS Puro al Amortiguamiento $\alpha$ (`tabla_sens_alpha` en `Red #1: trainbr`):**
  - $\alpha = 10^{-6}$ ($548$ podados, $112$ sin. activos): $E_{\text{val}} = 0.1227$, $\text{Err}_{\text{val}} = 23.33\%$, $\text{Err}_{\text{te\_int}} = 20.42\%$.
  - $\alpha = 10^{-5}$ ($576$ podados, $84$ sin. activos): $E_{\text{val}} = 0.1094$, $\text{Err}_{\text{val}} = 22.92\%$, $\text{Err}_{\text{te\_int}} = 21.25\%$.
  - $\alpha = 10^{-4}$ ($550$ podados, $110$ sin. activos): $E_{\text{val}} = 0.0724$, $\text{Err}_{\text{val}} = 18.33\%$, $\text{Err}_{\text{te\_int}} = 18.33\%$.
  - $\alpha = 10^{-3}$ ($553$ podados, $107$ sin. activos): $E_{\text{val}} = 0.0659$, $\text{Err}_{\text{val}} = 17.92\%$, $\text{Err}_{\text{te\_int}} = \mathbf{14.17\%}$ (cuando $\alpha$ coincide con $\lambda_{\text{eff}} = 10^{-3}$, OBS Puro sin reentrenamiento alcanza el mismo $14.17\%$ en Prueba Interna).
- **Comparativa de Poda OBS en las 3 Redes Finalistas (`resultados_top3_obs`):**
  1. **Red #1 (Campeona Seleccionada — `Candidata_1_trainbr`: `trainbr | Config B: tanh+lin`, Seed `2102`):**
     - **Pesos sinápticos eliminados:** $563$ de $660$ (**$85.30\%$ de compresión sináptica**), quedando **$97$ pesos sinápticos activos** ($82$ en $\mathbf{W}_1$ y $15$ en $\mathbf{W}_2$) y **$113 / 721$ parámetros totales activos** ($84.33\%$ compresión total).
     - **Neuronas ocultas activas:** De $60$ a **$15$ neuronas ocultas activas** ($\{10, 21, 23, 24, 28, 34, 35, 43, 44, 48, 49, 50, 53, 54, 57\}$, reducción del $75.0\%$).
     - **Desempeño:** $E_{\text{tr}} = 0.0573$ ($\text{Err}_{\text{tr}} = 15.62\%$), $E_{\text{val}} = 0.0652$ ($\text{Err}_{\text{val}} = 17.92\%$), y en Prueba Interna Hold-out mejora de $14.58\%$ a **$\mathbf{14.17\%}$** (**Exactitud $= 85.83\%$**, $\text{AUC}_{\text{te\_int}} = \mathbf{0.9252}$, $E_{\text{te\_int}} = 0.0570$).
     - **Selección implícita de características:** Mayor importancia en $x_9$ ($11/15$ conex., $\|\mathbf{W}_{1,(:,9)}\|_1 = 3.907$), $x_3$ ($10/15$, $3.468$), $x_1$ ($10/15$, $3.334$) y $x_2$ ($9/15$, $3.152$); menor importancia en las variables colineales/redundantes $x_6$ ($5/15$, $1.032$), $x_5$ ($6/15$, $1.253$), $x_7$ ($8/15$, $1.300$) y $x_4$ ($4/15$, $1.444$).
  2. **Red #2 (Finalista No-LM #2 — `trainrp | Config B: tanh+lin`, Seed `2504`):**
     - **Pesos sinápticos eliminados:** $579$ de $660$ (**$87.73\%$ de compresión sináptica**), quedando **$81$ pesos sinápticos activos**, **$16 / 60$ neuronas ocultas activas** y $98 / 721$ parámetros totales.
     - **Desempeño:** $E_{\text{tr}} = 0.0567$ ($\text{Err}_{\text{tr}} = 15.18\%$), $E_{\text{val}} = 0.0661$ ($\text{Err}_{\text{val}} = \mathbf{17.08\%}$), y en Prueba Interna Hold-out conserva intacto **$\mathbf{15.00\%}$** ($\text{AUC}_{\text{te\_int}} = 0.9163$).
  3. **Red #3 (Segunda Opción / Campeona en Compresión Extrema — `Candidata_2_trainlm`: `trainlm | Config C: sig+sig`, Seed `3603`):**
     - **Pesos sinápticos eliminados:** $601$ de $660$ (**$91.06\%$ de compresión sináptica**), quedando **$59$ pesos sinápticos activos** ($49$ en $\mathbf{W}_1$ y $10$ en $\mathbf{W}_2$), **$10 / 60$ neuronas ocultas activas** ($\{8, 15, 17, 23, 37, 39, 42, 47, 50, 51\}$, reducción del $83.3\%$) y **$70 / 721$ parámetros totales activos** ($90.29\%$ compresión total).
     - **Desempeño:** $E_{\text{tr}} = 0.0572$ ($\text{Err}_{\text{tr}} = 14.55\%$), alcanza el **menor MSE de Validación ($E_{\text{val}} = 0.0626$, $\text{Err}_{\text{val}} = 17.50\%$)** y el **menor MSE en Prueba Interna ($E_{\text{te\_int}} = 0.0558$)**, reduciendo su error de clasificación Hold-out de $16.25\%$ a **$\mathbf{15.83\%}$** ($\text{AUC}_{\text{te\_int}} = 0.9215$). Además, poda casi por completo la variable colineal $x_5$ (dejándole solo $1/10$ conexión activa con $\|\mathbf{W}_{1,(:,5)}\|_1 = 1.699$).
- **Intervalos de Confianza al 95% para el Error en el Conjunto Solución (`x_test`, $P = 13\,888$):**
  - **En `Candidata_1_trainbr` (Campeona Oficial):**
    - Hold-out ($P=240$): Puntual $= \mathbf{14.17\% \pm 2.25\%}$ (Exactitud $= \mathbf{85.83\%}$), IC 95% Wilson $= \mathbf{[10.32\%, 19.14\%]}$, IC 95% Bootstrap ($B=5000$) $= \mathbf{[10.00\%, 18.75\%]}$.
    - Fuera de muestra combinado ($P=480$): Puntual $= \mathbf{16.04\%}$ (Exactitud $= 83.96\%$), IC 95% Wilson $= \mathbf{[13.03\%, 19.59\%]}$, IC 95% Bootstrap $= \mathbf{[12.92\%, 19.38\%]}$.
  - **En `Candidata_2_trainlm` (Segunda Opción Hipercompacta):**
    - Hold-out ($P=240$): Puntual $= \mathbf{15.83\% \pm 2.36\%}$ (Exactitud $= 84.17\%$), IC 95% Wilson $= \mathbf{[11.76\%, 20.99\%]}$, IC 95% Bootstrap ($B=5000$) $= \mathbf{[11.25\%, 20.42\%]}$.
    - Fuera de muestra combinado ($P=480$): Puntual $= \mathbf{16.67\%}$, IC 95% Wilson $= \mathbf{[13.60\%, 20.26\%]}$.

### 3. Rutas de Scripts, Datos y Figuras Generadas
- **Script:** `03_Scripts/paso4_poda_obs.m`
- **Datos:** `05_Resultados_Parciales/poda_obs_punto4.mat`
- **Figuras:**
  - `04_Figuras/fig9_curvas_poda_obs_vs_obd_mag.png` (y copias específicas en `06_Entregables/Candidata_1_trainbr/` y `06_Entregables/Candidata_2_trainlm/`).
  - `04_Figuras/fig10_arquitectura_final_y_estimacion_error.png` (y copias específicas en `06_Entregables/Candidata_1_trainbr/` y `06_Entregables/Candidata_2_trainlm/`).

---

## Punto 5: Verificación del Efecto de Precisión de Punto Flotante, Simulación Comparativa en `x_test` y Empaquetado de Entregables (`OBS_30.MAT` y `pesos_30.mat`)

### 1. Ecuaciones y Fenómeno Numérico de Punto Flotante en MATLAB (Recomendación del Profesor)
Al evaluar en doble precisión IEEE-754 ($\epsilon_{\text{mach}} = 2.2204 \times 10^{-16}$) la ecuación de ajuste quirúrgico de Hassibi & Stork (Ec. 5):
$$\mathbf{w}_{\text{nuevo}} = \mathbf{w} - \frac{w_{q^*}}{[\mathbf{H}^{-1}]_{q^*q^*}}\mathbf{H}^{-1}\mathbf{e}_{q^*}$$
se demostraron experimentalmente **tres fallas numéricas críticas** en ambas redes candidatas cuando no se aplica la restricción explícita en cero y máscara:
1. **Cancelación inexacta directa (`a - (a/b)*b ~= 0`):** En **`71 / 660` pesos sinápticos ($10.8\%$)** de `Candidata_1_trainbr` (residuo máx. $5.55 \times 10^{-17}$) y en **`93 / 660` pesos sinápticos ($14.1\%$)** de `Candidata_2_trainlm` (residuo máx. $2.22 \times 10^{-16}$), la operación en punto flotante no da cero exacto, dejando residuos cuya saliencia en el paso siguiente cae a $L_{q^*} \sim 10^{-35}$.
2. **Estancamiento en un único peso (Modalidad 1A — Sin exclusión por máscara):** El peso eliminado en la iteración 1 ($q^* = 232$ en `trainbr`; $q^* = 554$ en `trainlm`) es elegido por `argmin(L)` durante las **50 iteraciones seguidas** (`1/50` pesos únicos eliminados).
3. **Resurrección acoplada o ciclo "Ping-Pong" (Modalidad 1B — Sin `w = w .* mask_act`):** Incluso si se ignora $w_q = 0$ en la búsqueda (`L(w==0) = inf`), el término fuera de la diagonal $[\mathbf{H}^{-1}]_{q_1, q_2} \neq 0$ en el paso 2 **resucita al peso ya eliminado en el paso 1** con magnitudes entre $10^{-11}$ y $10^{-150}$. El algoritmo queda atrapado infinitamente oscilando entre **2 únicos pesos** ($q^* \in \{232, 186\}$ en `trainbr` y $q^* \in \{554, 124\}$ en `trainlm`, eliminando solo `2/50` pesos en 50 pasos), mientras que con el **Triple Candado del Profesor** (`w(q_star)=0`, `mask_act(q_star)=false`, `w = w .* mask_act`, `L(~mask_act)=inf`) elimina **`50/50` pesos únicos** fijándolos exactamente en `0.0000000000000000`.

### 2. Resultados de la Simulación Comparativa sobre `x_test` ($P_{\text{test}} = 13\,888$) y Verificación de Entregables
Se ejecutó y auditó la simulación completa en dos carpetas independientes dentro de `06_Entregables/`:
- **Campeona Oficial (`06_Entregables/Candidata_1_trainbr/` y copias principales en `06_Entregables/` y raíz):**
  - Arquitectura efectiva `10 - 15 - 1` ($97$ pesos sinápticos activos, $113$ parámetros totales, activaciones `tanh + lineal`).
  - Predicciones en `x_test` ($13\,888 \times 1$): **$5\,623$ muestras Clase 0 ($40.49\%$)** y **$8\,265$ muestras Clase 1 ($59.51\%$)**.
  - Fidelidad con su red base de $660$ pesos: **$97.03\%$** ($13\,475 / 13\,888$ coinciden, $r = 0.9791$).
  - Acuerdo con el voto mayoritario (consenso de las 3 redes finalistas `trainbr`, `trainrp`, `trainlm`): **$97.54\%$**.
  - Equivalencia entre representación completa $10\text{-}60\text{-}1$ y compacta $10\text{-}15\text{-}1$: error continuo máximo $= 8.88 \times 10^{-16}$ ($100\%$ de etiquetas idénticas).
- **Segunda Opción Hipercompacta (`06_Entregables/Candidata_2_trainlm/`):**
  - Arquitectura efectiva `10 - 10 - 1` ($59$ pesos sinápticos activos, $70$ parámetros totales, activaciones `sigmoide + sigmoide`).
  - Predicciones en `x_test` ($13\,888 \times 1$): **$5\,932$ muestras Clase 0 ($42.71\%$)** y **$7\,956$ muestras Clase 1 ($57.29\%$)**, a solo **$0.43\%$** de la proporción teórica poblacional estimada por mezcla de medias en $x_9$ ($56.86\%$).
  - Fidelidad con su red base de $660$ pesos: **$95.87\%$** ($13\,314 / 13\,888$ coinciden, $r = 0.9864$).
  - Acuerdo con el consenso de las 3 redes finalistas: **$95.51\%$**.
  - Equivalencia entre representación completa $10\text{-}60\text{-}1$ y compacta $10\text{-}10\text{-}1$: error continuo máximo $= 4.44 \times 10^{-16}$ ($100\%$ de etiquetas idénticas).
- **Concordancia Cruzada Directa (`Candidata_1_trainbr` vs. `Candidata_2_trainlm` en `x_test`):**
  - Coinciden exactamente en **$12\,923 / 13\,888$ patrones ($93.05\%$ de acuerdo directo**, correlación continua $r = 0.9422$).

### 3. Rutas de Scripts, Entregables y Figuras Generadas
- **Script:** `03_Scripts/paso5_generar_entregables.m`
- **Datos Comparativos:** `05_Resultados_Parciales/comparativa_simulacion_paso5.mat`
- **Entregables Oficiales (Campeona `Candidata_1_trainbr`):** `06_Entregables/OBS_30.MAT`, `06_Entregables/pesos_30.mat` (más copias en raíz `OBS_30.MAT`, `pesos_30.mat` y subcarpeta `06_Entregables/Candidata_1_trainbr/`).
- **Entregables Segunda Opción (`Candidata_2_trainlm`):** `06_Entregables/Candidata_2_trainlm/OBS_30.MAT` y `06_Entregables/Candidata_2_trainlm/pesos_30.mat`.
- **Figuras:**
  - `04_Figuras/fig11_efecto_precision_matlab_candado.png` y `04_Figuras/fig12_simulacion_final_xtest_OBS30.png` (más los juegos completos `fig9`–`fig12` en `06_Entregables/Candidata_1_trainbr/` y `06_Entregables/Candidata_2_trainlm/`).
