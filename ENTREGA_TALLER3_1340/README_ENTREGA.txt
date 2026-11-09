========================================================================================
PAQUETE DE ENTREGA OFICIAL - TALLER NO. 3
DISENO Y PODA ESTRUCTURAL MEDIANTE OPTIMAL BRAIN SURGEON (OBS) ANALITICO PURO
========================================================================================

Asignatura: Inteligencia Computacional II
Profesor:   Andres Eduardo Gaona Barrera
Institucion: Universidad Distrital Francisco Jose de Caldas

Integrantes del Grupo:
- Maria Yuliany Sepulveda Bello - Codigo: 20202005013 (Ultimos digitos: 13)
- Martin Felipe Gil Quemba       - Codigo: 20221005040 (Ultimos digitos: 40)

Codigo Conjunto para Archivos de Entrega: 1340 (Sepulveda: 13 + Gil: 40)

========================================================================================
1. ESTRUCTURA DEL PAQUETE DE ENTREGA
========================================================================================

Este directorio contiene la totalidad de los archivos exigidos por el enunciado del
Taller 3, organizados de forma autocontenida:

[ARCHIVOS PRINCIPALES EN LA RAIZ]
- OBS_1340.MAT               : Vector [13888 x 1] double con las etiquetas binarias {0, 1}
                               calculadas para los patrones de prueba ciega x_test.
- pesos_1340.mat             : Estructura de MATLAB con los pesos y arquitectura final
                               reducida a 20 neuronas ocultas y 107 pesos sinapticos.
- verificar_reproducibilidad_entrega.m : Script de un solo clic en MATLAB que reproduce y
                               valida OBS_1340.MAT desde cero a partir de pesos_1340.mat
                               con 0 discrepancias sobre los 13,888 patrones.

[COPIAS DE COMPATIBILIDAD]
- OBS_13_40.mat / pesos_13_40.mat : Copias con guion bajo entre codigos estudiantiles.
- OBS_30.MAT    / pesos_30.mat    : Copias con la nomenclatura inicial.

[SUBDIRECTORIOS]
- 01_Scripts_MATLAB/         : Codigo fuente completo de las 5 etapas del proyecto.
    * woodbury_hessiana_inv.m         : Inversion de la Hessiana por recursion de Woodbury (Ec. 17).
    * ejecutar_poda_obs_pura.m        : Motor de poda OBS analitica pura (Ec. 5 y Schur).
    * analisis_exploratorio_taller3.m : Analisis estadistico exploratorio y separabilidad.
    * paso2_entrenar_red_base.m       : Benchmark multi-algoritmo (6 familias) y seleccion top 3.
    * paso3_hessiana_obs.m            : Calculo analitico de sensibilidades X_m y H^-1.
    * paso4_poda_obs.m                : Poda destructiva OBS puro en las 3 candidatas.
    * paso5_generar_entregables.m     : Evaluacion, 4 modalidades IEEE-754 y empaquetado.

- 02_Figuras/                : Las 12 figuras generadas en alta resolucion (200 dpi) para el informe.
    * fig1 a fig4 : Analisis exploratorio, correlaciones, PCA y distribucion de clases.
    * fig5 a fig6 : Benchmark de algoritmos y diagnostico de la red base campeona.
    * fig7 a fig8 : Mapas de la Hessiana, convergencia de Woodbury y sobreestimacion OBD.
    * fig9 a fig12: Curvas de poda OBS vs OBD vs Magnitud, arquitectura final y simulacion en x_test.

- 03_Informe_Tecnico/        : Informe formal completo del proyecto.
    * Informe_Taller3.tex : Codigo fuente en LaTeX (formato twocolumn letterpaper, neutralizado).
    * Informe_Taller3.txt : Informe tecnico completo en texto plano legible.

========================================================================================
2. RESUMEN TECNICO DE LA SOLUCION ENTREGADA
========================================================================================

- Arquitectura Inicial Base  : MLP 10-60-1 (721 parametros totales; 660 sinapsis + 61 sesgos).
- Algoritmo de Entrenamiento : Regularizacion Bayesiana de MacKay (trainbr, Config B: tanh+lin,
                               seed 2102, alpha_reg = 31.25, beta = 31250, gamma_eff = 126.3).
- Metodo de Poda             : Optimal Brain Surgeon (OBS) ANALITICO PURO segun Hassibi & Stork
                               (NIPS 1993, Ecs. 5 y 17) con CERO REENTRENAMIENTO.
- Formula de Inversion H^-1  : Recursion de Sherman-Morrison-Woodbury sobre productos externos
                               (Error relativo de Frobenius vs directa: 3.97e-12).
- Proteccion contra IEEE-754 : Triple candado numerico: L(~mask_act) = inf, w(q*) = 0 forzado,
                               y mascara booleana multiplicativa w = w .* mask_act.

[ARQUITECTURA FINAL OPTIMA SELECCIONADA]
- Pesos sinapticos eliminados: 553 de 660 (83.79% de compresion sinaptica).
- Neuronas ocultas activas   : 20 de 60 (66.7% de reduccion de capa oculta).
- Pesos sinapticos activos   : 107 (87 en W1 + 20 en W2).
- Parametros totales activos : 128 (107 sinapsis + 20 sesgos ocultos + 1 sesgo salida).

[METRICAS DE GENERALIZACION OBTENIDAS]
- Error en Prueba Interna Hold-Out (P = 240) : 14.17% (Exactitud = 85.83%, AUC = 0.9281)
                                               IC 95% Wilson: [10.32%, 19.14%]
- Error Fuera de Muestra Combinado (P = 480) : 16.04% (IC 95% Wilson: [13.03%, 19.59%])
- Prevalencia Estimada Clase 1 en x_test     : 57.37% (7,967 muestras predichas como Clase 1)
                                               Alineado con el 56.86% teorico de mezcla Gaussiana.
- Error Estimado Ajustado en x_test          : 16.30% (Ajuste epidemiologico de Rogan-Gladen,
                                               IC 95% Bootstrap: [13.03%, 19.96%]).

========================================================================================
3. INSTRUCCIONES DE VERIFICACION RAPIDA EN MATLAB
========================================================================================

Para verificar la integridad y reproducibilidad inmediata de los entregables:
1. Abra MATLAB y navegue a esta carpeta (ENTREGA_TALLER3_1340).
2. Asegurese de que el archivo 'datos_taller3.mat' este disponible en esta carpeta o en '../02_Datos/'.
3. Ejecute en la consola de comandos de MATLAB:
       >> verificar_reproducibilidad_entrega
4. El script cargara 'pesos_1340.mat', normalizara x_test con mu_x y sigma_x guardados,
   ejecutara la red reducida de 20 neuronas y 107 sinapsis y comparara contra 'OBS_1340.MAT'.
   El resultado comprobara 0 discrepancias (100.0000% de coincidencia exacta).
========================================================================================
