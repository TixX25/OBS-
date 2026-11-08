%% ANALISIS EXPLORATORIO DE DATOS - TALLER 3 (INTELIGENCIA COMPUTACIONAL II)
% Analisis del dataset datos_taller3.mat para el diseno de una red MLP 
% (10-60-1) con poda destructiva mediante Optimal Brain Surgeon (OBS).
% Referencia: Hassibi, B. & Stork, D.G. (1993). NIPS 5, pp. 164-171.

clear; clc; close all;

%% 1. Carga e inspeccion de la base de datos
base_dir = fileparts(fileparts(mfilename('fullpath')));
if isempty(base_dir), base_dir = pwd; end
fig_dir  = fullfile(base_dir, '04_Figuras');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

datos = load(fullfile(base_dir, '02_Datos', 'datos_taller3.mat'));
X_train = datos.x_train;
y_train = datos.y_train;
X_test  = datos.x_test;

[N_train, d] = size(X_train);
[N_test, ~]  = size(X_test);

fprintf('=================================================================\n');
fprintf('  RESUMEN GENERAL DE LA BASE DE DATOS (datos_taller3.mat)\n');
fprintf('=================================================================\n');
fprintf('Muestras de entrenamiento (x_train) : %d patrones x %d variables\n', N_train, d);
fprintf('Etiquetas de entrenamiento (y_train): %d etiquetas {0, 1}\n', length(y_train));
fprintf('Muestras de prueba (x_test)         : %d patrones x %d variables\n', N_test, d);
fprintf('Valores faltantes (NaN / Inf)       : %d en Train | %d en Test\n', ...
    sum(isnan(X_train(:)) | isinf(X_train(:))), sum(isnan(X_test(:)) | isinf(X_test(:))));

n_c0 = sum(y_train == 0);
n_c1 = sum(y_train == 1);
fprintf('Balance de clases en y_train        : Clase 0 = %d (%.1f%%) | Clase 1 = %d (%.1f%%)\n', ...
    n_c0, 100*n_c0/N_train, n_c1, 100*n_c1/N_train);

% Parametros de la arquitectura inicial solicitada en el Taller 3 (60 neuronas ocultas)
H_hidden = 60;
n_pesos_sin_bias = d * H_hidden + H_hidden * 1;
n_pesos_con_bias = (d + 1) * H_hidden + (H_hidden + 1) * 1;
fprintf('\n--- Relacion Capacidad de Red (10-%d-1) vs Datos (P = %d) ---\n', H_hidden, N_train);
fprintf('Numero de parametros (con biases)   : n = %d pesos\n', n_pesos_con_bias);
fprintf('Numero de parametros (sin biases)   : n = %d pesos\n', n_pesos_sin_bias);
fprintf('Ratio Muestras / Parametros (P / n) : %.2f (Cumple P > n para rango de H)\n\n', N_train / n_pesos_con_bias);

%% 2. Estadisticas descriptivas y poder discriminativo por variable
fprintf('========================================================================================\n');
fprintf('Var |   Min Tr  |   Max Tr  | Media C0 | Media C1 |  Std C0  |  Std C1  | Fisher J | KS Tr-Te\n');
fprintf('----------------------------------------------------------------------------------------\n');

fisher_J = zeros(d, 1);
ks_stat  = zeros(d, 1);
ks_pval  = zeros(d, 1);

for j = 1:d
    x0 = X_train(y_train == 0, j);
    x1 = X_train(y_train == 1, j);
    m0 = mean(x0); m1 = mean(x1);
    s0 = std(x0);  s1 = std(x1);
    
    % Indice discriminante de Fisher: (m0 - m1)^2 / (s0^2 + s1^2)
    fisher_J(j) = (m0 - m1)^2 / (s0^2 + s1^2);
    
    % Prueba de Kolmogorov-Smirnov de 2 muestras entre Train y Test
    [~, ks_pval(j), ks_stat(j)] = kstest2(X_train(:, j), X_test(:, j));
    
    fprintf(' x%-2d| %9.3f | %9.3f | %8.3f | %8.3f | %8.3f | %8.3f |  %6.4f  |  %6.4f\n', ...
        j, min(X_train(:,j)), max(X_train(:,j)), m0, m1, s0, s1, fisher_J(j), ks_stat(j));
end
fprintf('========================================================================================\n\n');

%% 3. FIGURA 1: Escalas, Rangos Dinamicos y Balance de Clases
fig1 = figure('Name', 'Escalas y Balance de Datos', 'Position', [100, 100, 1150, 480], 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel Izquierdo: Boxchart en escala original
nexttile;
boxchart(X_train);
grid on;
xticklabels(arrayfun(@(i) sprintf('x_{%d}', i), 1:d, 'UniformOutput', false));
title('Dispersión y Escala Original de las Variables (x\_train)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Variables de Entrada', 'FontSize', 11);
ylabel('Magnitud en Unidades Originales', 'FontSize', 11);
set(gca, 'FontSize', 10);

% Panel Derecho: Desviacion estandar por clase (evidencia de heterocedasticidad radial)
nexttile;
std_c0 = std(X_train(y_train == 0, :));
std_c1 = std(X_train(y_train == 1, :));
ratio_std = std_c0 ./ std_c1;
b = bar(1:d, [std_c0; std_c1]', 'grouped');
b(1).FaceColor = [0.8500 0.3250 0.0980];
b(2).FaceColor = [0.0000 0.4470 0.7410];
grid on;
xticks(1:d);
xticklabels(arrayfun(@(i) sprintf('x_{%d}', i), 1:d, 'UniformOutput', false));
title('Desviación Estándar por Clase (\sigma_{C0} > \sigma_{C1})', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Variables de Entrada', 'FontSize', 11);
ylabel('Desviación Estándar (\sigma)', 'FontSize', 11);
legend({'Clase 0 (Dispersa / Cola larga)', 'Clase 1 (Compacta / Núcleo)'}, 'Location', 'northwest', 'FontSize', 10);
set(gca, 'FontSize', 10);

exportgraphics(fig1, fullfile(fig_dir, 'fig1_escalas_y_dispersion.png'), 'Resolution', 200);

%% 4. FIGURA 2: Distribuciones por Clase (Clase 0 vs Clase 1) en las 10 Variables
fig2 = figure('Name', 'Distribuciones por Clase', 'Position', [80, 80, 1300, 600], 'Color', 'w');
tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

for j = 1:d
    nexttile;
    hold on;
    x0 = X_train(y_train == 0, j);
    x1 = X_train(y_train == 1, j);
    
    % Estimacion de densidad por kernel (KDE) o histogramas normalizados
    edges = linspace(min(X_train(:,j)), max(X_train(:,j)), 30);
    histogram(x0, edges, 'Normalization', 'pdf', 'FaceColor', [0.8500 0.3250 0.0980], ...
        'FaceAlpha', 0.45, 'EdgeColor', 'none');
    histogram(x1, edges, 'Normalization', 'pdf', 'FaceColor', [0.0000 0.4470 0.7410], ...
        'FaceAlpha', 0.45, 'EdgeColor', 'none');
    
    grid on;
    title(sprintf('Variable x_{%d} (J = %.3f)', j, fisher_J(j)), 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Valor', 'FontSize', 9);
    ylabel('Densidad', 'FontSize', 9);
    if j == 1
        legend({'Clase 0', 'Clase 1'}, 'Location', 'northeast', 'FontSize', 8);
    end
    hold off;
end
sgtitle('Distribuciones Condicionales por Clase en Entrenamiento (x\_train)', 'FontSize', 13, 'FontWeight', 'bold');
exportgraphics(fig2, fullfile(fig_dir, 'fig2_distribuciones_clases.png'), 'Resolution', 200);

%% 5. FIGURA 3: Matriz de Correlacion, Espectro de Autovalores y Colinealidad
fig3 = figure('Name', 'Correlacion y Colinealidad', 'Position', [100, 100, 1150, 480], 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel Izquierdo: Mapa de calor de la matriz de correlacion
nexttile;
R = corr(X_train);
imagesc(R);
colormap(gca, parula);
cb = colorbar;
cb.Label.String = 'Coeficiente de Correlación de Pearson (r)';
clim([-1 1]);
axis square;
xticks(1:d); yticks(1:d);
xticklabels(arrayfun(@(i) sprintf('x_{%d}', i), 1:d, 'UniformOutput', false));
yticklabels(arrayfun(@(i) sprintf('x_{%d}', i), 1:d, 'UniformOutput', false));
title('Matriz de Correlación de Entradas (x\_train)', 'FontSize', 12, 'FontWeight', 'bold');

% Anotar valores numericos en la matriz
for r_idx = 1:d
    for c_idx = 1:d
        val = R(r_idx, c_idx);
        txt_col = 'k';
        if abs(val) < 0.4, txt_col = 'w'; end
        text(c_idx, r_idx, sprintf('%.2f', val), 'HorizontalAlignment', 'center', ...
            'FontSize', 8, 'FontWeight', 'bold', 'Color', txt_col);
    end
end

% Panel Derecho: Varianza explicada acumulada (PCA) e impacto en la Hessiana
nexttile;
X_std = zscore(X_train);
[coeff, score, latent, ~, explained] = pca(X_std);
cum_exp = cumsum(explained);

yyaxis left;
bar(1:d, latent, 0.6, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.7);
ylabel('Autovalor de Correlación (\lambda_i)', 'FontSize', 11);
ylim([0 max(latent)*1.15]);

yyaxis right;
plot(1:d, cum_exp, '-o', 'LineWidth', 2.2, 'MarkerSize', 6, 'Color', [0.85 0.2 0.1], 'MarkerFaceColor', [0.85 0.2 0.1]);
ylabel('Varianza Explicada Acumulada (%)', 'FontSize', 11);
ylim([0 105]);

grid on;
xticks(1:d);
xlabel('Componente Principal (PC)', 'FontSize', 11);
title(sprintf('Espectro PCA (\\kappa(R) = %.1f, \\lambda_{10} = %.3f)', max(latent)/min(latent), min(latent)), ...
    'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);

exportgraphics(fig3, fullfile(fig_dir, 'fig3_correlacion_y_pca.png'), 'Resolution', 200);

%% 6. FIGURA 4: Geometria No Lineal de las Clases y Consistencia Train vs Test
fig4 = figure('Name', 'Geometria y Consistencia Train-Test', 'Position', [100, 100, 1150, 480], 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel Izquierdo: Espacio de las variables mas informativas y no lineales (x9, x1, x6)
nexttile;
hold on;
idx0 = (y_train == 0);
idx1 = (y_train == 1);
scatter(X_train(idx0, 9), X_train(idx0, 6), 22, [0.8500 0.3250 0.0980], 'filled', 'MarkerFaceAlpha', 0.55);
scatter(X_train(idx1, 9), X_train(idx1, 6), 22, [0.0000 0.4470 0.7410], 'filled', 'MarkerFaceAlpha', 0.65);
grid on;
xlabel('Variable x_9 (Mayor poder discriminante lineal, J = 0.534)', 'FontSize', 10.5);
ylabel('Variable x_6 (Diferencia de varianza: \sigma_{C0}=84.6 vs \sigma_{C1}=38.9)', 'FontSize', 10.5);
title('Estructura No Lineal: Núcleo Compacto (C1) vs Halo (C0)', 'FontSize', 11.5, 'FontWeight', 'bold');
legend({'Clase 0 (Halo disperso)', 'Clase 1 (Núcleo concentrado)'}, 'Location', 'northeast', 'FontSize', 9.5);
hold off;

% Panel Derecho: Comparacion de funciones de distribucion acumulada (CDF) Train vs Test
nexttile;
hold on;
vars_check = [1, 4, 6, 9];
colors = lines(length(vars_check));
h_lines = gobjects(length(vars_check)*2, 1);

for k = 1:length(vars_check)
    v = vars_check(k);
    % Estandarizar con media y std de train para superponer en un mismo eje
    mu_v = mean(X_train(:,v));
    sd_v = std(X_train(:,v));
    
    [f_tr, x_tr_cdf] = ecdf((X_train(:,v) - mu_v) / sd_v);
    [f_te, x_te_cdf] = ecdf((X_test(:,v)  - mu_v) / sd_v);
    
    h_lines(2*k-1) = plot(x_tr_cdf, f_tr, '-',  'Color', colors(k,:), 'LineWidth', 1.8, ...
        'DisplayName', sprintf('x_{%d} Train', v));
    h_lines(2*k)   = plot(x_te_cdf, f_te, '--', 'Color', colors(k,:), 'LineWidth', 1.8, ...
        'DisplayName', sprintf('x_{%d} Test (KS=%.3f)', v, ks_stat(v)));
end
grid on;
xlim([-3.5 3.5]);
xlabel('Valor Estandarizado (z-score respecto a Train)', 'FontSize', 10.5);
ylabel('Probabilidad Acumulada F(z)', 'FontSize', 10.5);
title('Alineación de Distribuciones: Entrenamiento vs Prueba', 'FontSize', 11.5, 'FontWeight', 'bold');
legend(h_lines, 'Location', 'southeast', 'FontSize', 8.5, 'NumColumns', 2);
hold off;

exportgraphics(fig4, fullfile(fig_dir, 'fig4_geometria_y_traintest.png'), 'Resolution', 200);
fprintf('Figuras exportadas exitosamente en la carpeta del taller.\n');
