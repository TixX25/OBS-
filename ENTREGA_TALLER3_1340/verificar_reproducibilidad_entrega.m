%% VERIFICACION DE REPRODUCIBILIDAD - TALLER 3 (OBS)
% Autores: Maria Yuliany Sepulveda Bello (20202005013) y Martin Felipe Gil Quemba (20221005040)
% Inteligencia Computacional II - Profesor: Andres Eduardo Gaona Barrera
% Universidad Distrital Francisco Jose de Caldas

clear; clc;

fprintf('=========================================================================\n');
fprintf('  VERIFICACION DE REPRODUCIBILIDAD: ENTREGABLES OFICIALES OBS_1340\n');
fprintf('=========================================================================\n\n');

% 1. Localizar archivo de datos original
datos_file = fullfile('..', '02_Datos', 'datos_taller3.mat');
if ~exist(datos_file, 'file')
    datos_file = 'datos_taller3.mat';
end
if ~exist(datos_file, 'file')
    error('Por favor coloque el archivo datos_taller3.mat en esta carpeta o en ../02_Datos/');
end

% 2. Cargar entregables
d_raw   = load(datos_file);
p_pesos = load('pesos_1340.mat');
d_obs   = load('OBS_1340.MAT');

fprintf('1. Carga de Archivos:\n');
fprintf('   - datos_taller3.mat : x_test = [%d x %d]\n', size(d_raw.x_test,1), size(d_raw.x_test,2));
fprintf('   - pesos_1340.mat    : W1 = [%d x %d], W2 = [%d x %d]\n', ...
    size(p_pesos.W1_compacta,1), size(p_pesos.W1_compacta,2), ...
    size(p_pesos.W2_compacta,1), size(p_pesos.W2_compacta,2));
fprintf('   - OBS_1340.MAT      : vector = [%d x %d]\n\n', size(d_obs.OBS_1340,1), size(d_obs.OBS_1340,2));

% 3. Estandarizacion z-score usando exclusivamente mu_x y sigma_x guardados
X_test_norm = (d_raw.x_test - p_pesos.mu_x) ./ p_pesos.sigma_x;

% 4. Evaluacion directa (Forward Pass)
Z1 = p_pesos.W1_compacta * X_test_norm' + p_pesos.b1_compacta;
if strcmp(p_pesos.act_hid, 'tanh')
    A1 = tanh(Z1);
else
    A1 = 1 ./ (1 + exp(-Z1));
end

z2 = p_pesos.W2_compacta * A1 + p_pesos.b2_compacta;
if strcmp(p_pesos.act_out, 'sig')
    o_test = 1 ./ (1 + exp(-z2));
else
    o_test = z2;
end
y_pred_indep = double(o_test' >= 0.5);

% 5. Comparacion exacta contra OBS_1340.MAT
N = length(y_pred_indep);
discrepancias = sum(y_pred_indep ~= d_obs.OBS_1340);
max_dif = max(abs(y_pred_indep - d_obs.OBS_1340));

fprintf('2. Resultados de la Verificacion:\n');
fprintf('   - Total de patrones simulados    : %d\n', N);
fprintf('   - Discrepancias con OBS_1340.MAT : %d (%.4f%%)\n', discrepancias, 100*discrepancias/N);
fprintf('   - Maxima diferencia binaria      : %g\n', max_dif);
fprintf('   - Distribucion de predicciones   : Clase 0 = %d (%.2f%%) | Clase 1 = %d (%.2f%%)\n\n', ...
    sum(y_pred_indep == 0), 100*mean(y_pred_indep == 0), ...
    sum(y_pred_indep == 1), 100*mean(y_pred_indep == 1));

if discrepancias == 0
    fprintf('=========================================================================\n');
    fprintf('  VALIDACION EXITOSA: REPRODUCCION EXACTA AL 100.0000%% (0 ERRORES)\n');
    fprintf('=========================================================================\n');
else
    fprintf('=========================================================================\n');
    fprintf('  ADVERTENCIA: SE DETECTARON DISCREPANCIAS.\n');
    fprintf('=========================================================================\n');
end
