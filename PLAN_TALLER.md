# Plan de Trabajo — Taller 3: Poda de Redes Neuronales con Optimal Brain Surgeon (OBS)

**Asignatura:** Inteligencia Computacional II  
**Referencias Base:**  
1. Hassibi, B. & Stork, D. G. (1993). *Second order derivatives for network pruning: Optimal Brain Surgeon*, NIPS 5, pp. 164–171 (`01_Documentacion/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon.pdf`).  
2. Guía de estudio del artículo (`01_Documentacion/OBS_guia_de_estudio.pdf`).  
3. Enunciado oficial (`01_Documentacion/TallerIII.pdf`) y dataset (`02_Datos/datos_taller3.mat`, con $P_{\text{total}} = 1600$ patrones de entrenamiento de $d = 10$ entradas y salida binaria $\{0, 1\}$, y $P_{\text{test}} = 13\,888$ patrones de prueba).

---

## Estructura de Carpetas del Proyecto

Todo el trabajo se organiza de forma modular en las siguientes rutas dentro del workspace:
- `01_Documentacion/`: Enunciado del taller, paper original de Hassibi & Stork (1993) y guía de estudio de OBS.
- `02_Datos/`: Base de datos original (`datos_taller3.mat`) y particiones estratificadas preprocesadas.
- `03_Scripts/`: Scripts y funciones de MATLAB organizados por etapa (`analisis_exploratorio_taller3.m`, `paso2_entrenar_red_base.m`, etc.).
- `04_Figuras/`: Todas las gráficas exportadas en alta resolución (`.png`) para el análisis y el informe final.
- `05_Resultados_Parciales/`: Archivos `.mat` intermedios (redes candidatas multi-arranque, mejor red base entrenada, matrices $\mathbf{H}$ y $\mathbf{H}^{-1}$, historial de poda).
- `06_Entregables/`: Archivos finales listos para subir al aula virtual (`OBS_codigo.MAT` y `pesos_codigo.mat`).

---

## Marco Teórico y Reglas de Implementación Acordadas

1. **Arquitectura Base:** Perceptrón Multicapa (MLP) de tres capas con estructura `10 - 60 - 1` (10 entradas, 1 capa oculta con $n_j = 60$ neuronas, 1 neurona de salida).
   - Número total de parámetros con biases: $n = (10 \times 60 + 60) + (60 \times 1 + 1) = 721$ pesos ($660$ pesos sinápticos sin biases).
2. **Búsqueda Exhaustiva Multi-Semilla y Multi-Configuración (Criterio de Máximo Desempeño):**
   - Como el problema de optimización no lineal tiene múltiples mínimos locales y el **50% de la nota depende de la efectividad en `x_test`**, cada etapa crítica se ejecutará mediante **experimentos multi-arranque (múltiples semillas aleatorias, funciones de activación y hiperparámetros de regularización $\lambda_{\text{reg}}$ y amortiguamiento $\alpha$)**.
   - Se reportarán estadísticas de **media, desviación estándar y mejor modelo**, seleccionando siempre el campeón según el desempeño de generalización.
3. **Ecuaciones de Sensibilidad (Gradientes de la Salida respecto a los Pesos, Ecs. 19–21 del paper):**
   Para cada patrón de entrenamiento $k \in \{1, \dots, P\}$, sea $o^{[k]} = F(\mathbf{w}, \mathbf{in}^{[k]})$ la salida de la red:
   $$\mathbf{X}^{[k]} = \frac{\partial F(\mathbf{w}, \mathbf{in}^{[k]})}{\partial \mathbf{w}} = \begin{pmatrix} \mathbf{X}_v^{[k]} \\ \mathbf{X}_u^{[k]} \end{pmatrix}$$
   - Para los pesos oculta $\to$ salida $v_j$:
     $$X_{v_j}^{[k]} = f_{\text{out}}'(\text{net}_{\text{out}}^{[k]}) \cdot o_j^{[k]}$$
   - Para los pesos entrada $\to$ oculta $u_{ji}$:
     $$X_{u_{ji}}^{[k]} = f_{\text{out}}'(\text{net}_{\text{out}}^{[k]}) \cdot f_{\text{hid}}'(\text{net}_j^{[k]}) \cdot v_j \cdot o_i^{[k]}$$
4. **Inversa de la Matriz Hessiana por Productos Externos (Recursión de Woodbury, Ec. 17 del paper):**
   $$\mathbf{H}_{m+1}^{-1} = \mathbf{H}_m^{-1} - \frac{\mathbf{H}_m^{-1} \mathbf{X}^{[m+1]} (\mathbf{X}^{[m+1]})^T \mathbf{H}_m^{-1}}{P + (\mathbf{X}^{[m+1]})^T \mathbf{H}_m^{-1} \mathbf{X}^{[m+1]}}, \qquad \text{con } \mathbf{H}_0^{-1} = \alpha^{-1}\mathbf{I}, \quad \alpha \in [10^{-8}, 10^{-4}]$$
5. **Saliencia, Ajuste Cirujano y Restricción Estricta en Cero (Ec. 5 del paper + Recomendación del Profesor):**
   - Saliencia de cada peso activo ($q \in \text{activos}$):
     $$L_q = \frac{w_q^2}{2 [\mathbf{H}^{-1}]_{qq}}$$
   - Selección del peso a eliminar: $q^* = \arg\min_{q \in \text{activos}} L_q$.
   - Actualización de todos los pesos:
     $$\mathbf{w} \leftarrow \mathbf{w} - \frac{w_{q^*}}{[\mathbf{H}^{-1}]_{q^*q^*}} \mathbf{H}^{-1} \mathbf{e}_{q^*}$$
   - **Restricción numérica obligatoria (indicación del profesor):** Para evitar residuos de punto flotante en MATLAB (ej. `1e-16` o `1e-7`) que harían que el algoritmo intente podar repetidamente el mismo peso:
     1. Se fuerza explícitamente `w(q_star) = 0;`.
     2. Se actualiza un vector máscara booleano `mask_activos(q_star) = false;` y se multiplica `w = w .* mask_activos;` en cada iteración (y durante cualquier etapa de ajuste/reentrenamiento).
     3. En la evaluación de $L_q$, los pesos con `mask_activos == false` (y los biases cuando se protegen) reciben $L_q = +\infty$.

---

## Hoja de Ruta Secuencial por Puntos

### Punto 1: Análisis Exploratorio de la Base de Datos (`datos_taller3.mat`) y Organización del Proyecto
- **Objetivo y ecuaciones base:** Caracterizar dimensionalidad ($P=1600, d=10, P_{\text{test}}=13\,888$), balance de clases, escalas dinámicas, índice discriminante de Fisher $J = \frac{(\mu_0 - \mu_1)^2}{\sigma_0^2 + \sigma_1^2}$, estructura de correlación $\mathbf{R}$ (colinealidad entre variables que produce una Hessiana $\mathbf{H}$ fuertemente no-diagonal) y prueba de Kolmogorov-Smirnov entre entrenamiento y prueba. Organizar todos los archivos en carpetas temáticas.
- **Lenguaje y ruta destino:** MATLAB (`03_Scripts/analisis_exploratorio_taller3.m` y figuras en `04_Figuras/`).
- **Criterio de validación:** Tablas estadísticas en consola y 4 figuras exportadas (`04_Figuras/fig1_escalas_y_dispersion.png` a `04_Figuras/fig4_geometria_y_traintest.png`).
- **Estado:** `[x] Completado`

---

### Punto 2: Partición Estratificada, Estandarización y Benchmark Multi-Algoritmo de la Red Base (MLP 10-60-1)
- **Objetivo y ecuaciones base:**
  1. Dividir `x_train` y `y_train` ($1600$ muestras) de forma estratificada en **Entrenamiento** ($70\% = 1120$ muestras, cumpliendo $P_{\text{train}} = 1120 > n = 721$), **Validación** ($15\% = 240$ muestras) y **Prueba Interna / Hold-out** ($15\% = 240$ muestras), aplicando estandarización *z-score* ($\tilde{x}_i = \frac{x_i - \mu_{\text{tr}, i}}{\sigma_{\text{tr}, i}}$) libre de fuga de datos.
  2. **Benchmark Exhaustivo Multi-Algoritmo (6 Familias de Entrenamiento $\times$ 3 Configuraciones de Activación):** Evaluar sistemáticamente:
     - **Regularización Bayesiana de MacKay (`trainbr`):** Minimiza $F(\mathbf{w}) = \beta E_D(\mathbf{w}) + \alpha E_W(\mathbf{w})$ estimando dinámicamente $\gamma_{\text{eff}} = \sum_{i=1}^n \frac{\lambda_i}{\lambda_i + \alpha/\beta}$ (MacKay, 1992; Foresee & Hagan, 1997).
     - **Cuasi-Newton BFGS / L-BFGS (`trainbfg`):** Actualización secante de rango 2 de la inversa de la Hessiana con búsqueda de línea de Armijo.
     - **Gradiente Conjugado Escalado (`trainscg`):** Algoritmo de segundo orden con región de confianza de Møller (1993).
     - **Gradiente Conjugado de Polak-Ribière (`traincgp`):** Direcciones conjugadas con reinicio automático.
     - **Resilient Backpropagation (`trainrp`):** Paso adaptativo por signo de coordenada con pulido local de curvatura.
     - **Levenberg-Marquardt (`trainlm`):** Optimización Gauss-Newton amortiguada $(\mathbf{H}_{\text{GN}} + \mu \mathbf{I})^{-1}\mathbf{g}$.
  3. **Selección de las 3 Redes Base Finalistas (`top3_modelos`) para Aplicar el Algoritmo OBS:**
     - **Red 1 (Mejor No-LM #1 / Campeona Global):** Mejor red de Regularización Bayesiana (`trainbr`).
     - **Red 2 (Mejor No-LM #2):** Segunda mejor red entre las familias que no son LM (`trainbfg` / `trainscg`).
     - **Red 3 (Mejor LM):** Mejor red entrenada con Levenberg-Marquardt (`trainlm`).
- **Lenguaje y ruta destino:** MATLAB (`03_Scripts/paso2_entrenar_red_base.m`, resultados en `05_Resultados_Parciales/red_base_10_60_1.mat`, figuras en `04_Figuras/`).
- **Criterio de validación:** Tabla comparativa completa por familia algorítmica y configuración de activación (media $\pm$ DE y mínimo en MSE, Error %, $\|\mathbf{w}\|_2$, $\gamma_{\text{eff}}$ y $\|\nabla E\|_2$), selección de las 3 redes finalistas (`2 No-LM + 1 LM`), curvas de convergencia, distribución de pesos iniciales y matriz de confusión/ROC.
- **Estado:** `[x] Completado`

---

### Punto 3: Implementación del Motor Matemático de $\mathbf{H}^{-1}$ por Productos Externos (Sobre las 3 Redes Seleccionadas)
- **Objetivo y ecuaciones base:**
  1. Programar el cálculo exacto del vector de derivadas de la salida respecto a los pesos $\mathbf{X}^{[k]} = \frac{\partial F(\mathbf{w}, \mathbf{in}^{[k]})}{\partial \mathbf{w}} \in \mathbb{R}^{n \times 1}$ (Ecs. 19–21 del paper) cuidando el ordenamiento lexicográfico entre matrices de pesos y el vector $\mathbf{w}$.
  2. Implementar la recursión de Woodbury (Ec. 17 del paper) para construir $\mathbf{H}^{-1}$ paso a paso sobre los patrones de entrenamiento y explorar la sensibilidad al parámetro $\alpha \in [10^{-8}, 10^{-2}]$ en las **3 redes seleccionadas** (las 2 mejores No-LM y la mejor LM).
  3. Verificar numéricamente las derivadas analíticas $\mathbf{X}^{[k]}$ contra diferencias finitas centradas, visualizar el mapa de calor de $\mathbf{H}$ y $\mathbf{H}^{-1}$ (demostrando que $\mathbf{H}$ es fuertemente no-diagonal) y comparar el condicionamiento espectral y las saliencias iniciales de las 3 redes.
- **Lenguaje y ruta destino:** MATLAB (`03_Scripts/paso3_hessiana_obs.m`, figuras en `04_Figuras/`, datos en `05_Resultados_Parciales/`).
- **Criterio de validación:** Error relativo entre gradiente analítico $\mathbf{X}^{[k]}$ y numérico $< 10^{-8}$, error de inversión de Woodbury frente a inversión directa $(\mathbf{H} + \alpha\mathbf{I})^{-1}$ del orden de precisión de máquina, y figuras de mapas de calor y diagnóstico de un paso de poda.
- **Estado:** `[x] Completado`

---

### Punto 4: Algoritmo de Poda Optimal Brain Surgeon (OBS) sobre las 3 Redes Finalistas (2 No-LM vs 1 LM) y Selección de la Red Óptima
- **Objetivo y ecuaciones base:**
  1. Ejecutar el ciclo destructivo de OBS aplicando la ecuación de saliencia $L_q = \frac{w_q^2}{2[\mathbf{H}^{-1}]_{qq}}$ y la actualización de pesos $\delta \mathbf{w} = -\frac{w_q}{[\mathbf{H}^{-1}]_{qq}}\mathbf{H}^{-1}\mathbf{e}_q$ (Ec. 5).
  2. Aplicar en cada corte la **restricción estricta recomendada por el profesor**: fijar `w(q_star) = 0`, anular la entrada en `mask_activos(q_star) = false`, aplicar `w = w .* mask_activos`, y asignar `L(~mask_activos) = inf` para que jamás intente volver a podar un peso eliminado.
  3. **Comparativa directa de Poda OBS entre las 3 Redes Seleccionadas (las 2 mejores No-LM y la mejor LM) y frente a OBD y Magnitud:**
     - Podar iterativamente las 3 redes seleccionadas del Punto 2 (evaluando tanto OBS con recálculo adaptativo de $\mathbf{H}^{-1}$ como OBS con pulido en subespacio con máscara fija `w = w .* mask_activos`) y contrastar su comportamiento frente a **Poda por Magnitud** y **Optimal Brain Damage (OBD)**.
  4. Identificar el **tamaño apropiado de la red** que minimiza el error de generalización en Validación y calcular el **estimativo riguroso del error sobre el conjunto solución** (intervalos de confianza de Wilson y Bootstrap al 95%) usando el conjunto de Prueba Interna.
- **Lenguaje y ruta destino:** MATLAB (`03_Scripts/paso4_poda_obs.m`, resultados en `05_Resultados_Parciales/`, figuras en `04_Figuras/`).
- **Criterio de validación:** Curvas comparativas de MSE y Error de Clasificación (%) vs. número de pesos activos para las 3 redes (2 No-LM y 1 LM) y para OBS vs. OBD vs. Magnitud; visualización de la arquitectura podada final y tabla comparativa completa.
- **Estado:** `[x] Completado`

---

### Punto 5: Verificación del Efecto de Precisión de Punto Flotante (Con vs. Sin Restricción en Cero), Simulación Final sobre `x_test` y Empaquetado de Entregables (`OBS_30.MAT` y `pesos_30.mat`)
- **Objetivo y ecuaciones base:**
  1. **Demostración experimental de la recomendación del profesor (Precisión IEEE-754 en MATLAB):** Ejecutar la prueba controlada del bucle OBS con vs. sin forzado a cero y máscara booleana sobre las redes candidatas (`Candidata_1_trainbr` y `Candidata_2_trainlm`).
  2. Evaluar y comparar en carpetas separadas (`06_Entregables/Candidata_1_trainbr/` y `06_Entregables/Candidata_2_trainlm/`) las dos arquitecturas campeonas podadas sobre los $13\,888$ patrones de `x_test`, contrastando su concordancia cruzada y con el consenso de las 3 redes finalistas.
  3. Guardar en `06_Entregables/` (y raíz) los archivos oficiales de la campeona seleccionada (`Candidata_1_trainbr`: `OBS_30.MAT` y `pesos_30.mat`), y actualizar `BITACORA_RESULTADOS.md` e `Informe_Taller3.tex` documentando la comparativa de las 3 redes finalistas y el análisis detallado tanto de la Campeona (`trainbr`) como de la Segunda Opción hipercompacta (`trainlm`).
- **Lenguaje y ruta destino:** MATLAB (`03_Scripts/paso5_generar_entregables.m`, figuras en `04_Figuras/` y archivos finales en `06_Entregables/`).
- **Criterio de validación:** Demostración numérica y gráfica del fenómeno de punto flotante; verificación de dimensiones (`13888 x 1`), ausencia de `NaN`, valores estrictamente en `{0, 1}`, y comprobación de reproducibilidad cargando `pesos_30.mat` desde cero para regenerar `OBS_30.MAT` en ambas carpetas.
- **Estado:** `[x] Completado`
