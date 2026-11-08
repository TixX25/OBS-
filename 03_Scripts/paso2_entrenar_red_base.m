%% PASO 2: PARTICION ESTRATIFICADA, ESTANDARIZACION Y BENCHMARK DE 6 ALGORITMOS (MLP 10-60-1)
% Evaluacion multi-arranque de 6 familias de algoritmos de entrenamiento (trainbr, trainbfg,
% trainscg, traincgp, trainrp y trainlm) en 3 configuraciones de activacion para poda
% mediante Optimal Brain Surgeon (Hassibi & Stork, NIPS 1993, Ecs. 1-7).

clear; clc; close all;

%% 1. Rutas y carga de datos originales
base_dir = fileparts(fileparts(mfilename('fullpath')));
if isempty(base_dir), base_dir = pwd; end
datos_dir = fullfile(base_dir, '02_Datos');
fig_dir   = fullfile(base_dir, '04_Figuras');
res_dir   = fullfile(base_dir, '05_Resultados_Parciales');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

raw = load(fullfile(datos_dir, 'datos_taller3.mat'));
x_train_raw = raw.x_train; y_train_raw = raw.y_train; x_test_raw = raw.x_test;
[~, d] = size(x_train_raw);
M = 60; n = (d + 1)*M + (M + 1)*1; % n = 721 parametros con biases

%% 2. Particion estratificada reproducible (70% Train = 1120, 15% Val = 240, 15% Test_Int = 240)
rng(42, 'twister');
idx0 = find(y_train_raw == 0); idx1 = find(y_train_raw == 1);
p0 = idx0(randperm(length(idx0))); p1 = idx1(randperm(length(idx1)));

idx_tr     = [p0(1:560);   p1(1:560)];
idx_val    = [p0(561:680); p1(561:680)];
idx_te_int = [p0(681:800); p1(681:800)];

idx_tr     = idx_tr(randperm(length(idx_tr)));
idx_val    = idx_val(randperm(length(idx_val)));
idx_te_int = idx_te_int(randperm(length(idx_te_int)));

assert(isempty(intersect(idx_tr, idx_val)) && isempty(intersect(idx_tr, idx_te_int)) && ...
       isempty(intersect(idx_val, idx_te_int)), 'Error: Solapamiento entre particiones');
assert(length(idx_tr) == 1120 && length(idx_val) == 240 && length(idx_te_int) == 240, 'Error de tamanos');

%% 3. Estandarizacion z-score usando exclusivamente estadisticos de Entrenamiento
mu_x    = mean(x_train_raw(idx_tr, :), 1);
sigma_x = std(x_train_raw(idx_tr, :), 0, 1);

X_tr        = (x_train_raw(idx_tr, :)     - mu_x) ./ sigma_x; y_tr     = y_train_raw(idx_tr);
X_val       = (x_train_raw(idx_val, :)    - mu_x) ./ sigma_x; y_val    = y_train_raw(idx_val);
X_te_int    = (x_train_raw(idx_te_int, :) - mu_x) ./ sigma_x; y_te_int = y_train_raw(idx_te_int);
X_test_norm = (x_test_raw                 - mu_x) ./ sigma_x;

assert(max(abs(mean(X_tr, 1))) < 1e-14 && max(abs(std(X_tr, 0, 1) - 1)) < 1e-14, 'Error en z-score');

save(fullfile(datos_dir, 'particion_datos.mat'), ...
    'X_tr', 'y_tr', 'X_val', 'y_val', 'X_te_int', 'y_te_int', ...
    'X_test_norm', 'mu_x', 'sigma_x', 'idx_tr', 'idx_val', 'idx_te_int');

fprintf('=========================================================================================\n');
fprintf('  PASO 2: BENCHMARK DE 6 ALGORITMOS DE ENTRENAMIENTO PARA RED BASE (MLP 10-60-1, n=721)\n');
fprintf('=========================================================================================\n');
fprintf('Particion (rng=42) : Train=%d (C0:%d, C1:%d) | Val=%d (C0:%d, C1:%d) | Test_Int=%d (C0:%d, C1:%d)\n', ...
    length(y_tr), sum(y_tr==0), sum(y_tr==1), length(y_val), sum(y_val==0), sum(y_val==1), ...
    length(y_te_int), sum(y_te_int==0), sum(y_te_int==1));
fprintf('Condicion de rango : P_tr = %d > n = %d (Ratio P_tr/n = %.2f)\n\n', length(y_tr), n, length(y_tr)/n);

%% 4. Verificacion del gradiente analitico vs diferencias finitas centradas
rng(999, 'twister');
w_chk = 0.2 * randn(n, 1);
[~, g_ana] = eval_mlp_loss_grad(w_chk, X_tr, y_tr, d, M, 1e-4, 'tanh', 'sig');
eps_fd = 1e-6; idx_chk = [1:10, 601:605, 661:665, 721];
g_num = zeros(size(idx_chk));
for ii = 1:length(idx_chk)
    k_idx = idx_chk(ii);
    wp = w_chk; wp(k_idx) = wp(k_idx) + eps_fd;
    wm = w_chk; wm(k_idx) = wm(k_idx) - eps_fd;
    Jp = eval_mlp_loss_grad(wp, X_tr, y_tr, d, M, 1e-4, 'tanh', 'sig');
    Jm = eval_mlp_loss_grad(wm, X_tr, y_tr, d, M, 1e-4, 'tanh', 'sig');
    g_num(ii) = (Jp - Jm) / (2 * eps_fd);
end
err_grad_rel = max(abs(g_ana(idx_chk)' - g_num)) / max(abs(g_ana(idx_chk)));
fprintf('Verificacion Gradiente Analitico vs Diferencias Finitas: Error Relativo Max = %.2e\n\n', err_grad_rel);
assert(err_grad_rel < 1e-8, 'Error: Gradiente analitico no coincide con diferencias finitas');

%% 5. Definicion de Configuraciones de Activacion y 6 Familias de Algoritmos
configs = {
    'tanh', 'sig', 'Config A (tanh+sig)';
    'tanh', 'lin', 'Config B (tanh+lin)';
    'sig',  'sig', 'Config C (sig+sig)'
};
n_conf = size(configs, 1);

% Grilla de 13 combinaciones (algoritmo + hiperparametro) por cada configuracion de activacion
algo_specs = {
    1, 'trainbr',  0.50, 'trainbr (Bayesiana Suave)';
    1, 'trainbr',  1.20, 'trainbr (Bayesiana MacKay)';
    2, 'trainbfg', 0.0,  'trainbfg (BFGS lam=0)';
    2, 'trainbfg', 1e-4, 'trainbfg (BFGS lam=1e-4)';
    2, 'trainbfg', 5e-4, 'trainbfg (BFGS lam=5e-4)';
    3, 'trainscg', 1e-4, 'trainscg (SCG lam=1e-4)';
    3, 'trainscg', 5e-4, 'trainscg (SCG lam=5e-4)';
    4, 'traincgp', 1e-4, 'traincgp (CG-PR lam=1e-4)';
    4, 'traincgp', 5e-4, 'traincgp (CG-PR lam=5e-4)';
    5, 'trainrp',  1e-4, 'trainrp (Rprop lam=1e-4)';
    5, 'trainrp',  5e-4, 'trainrp (Rprop lam=5e-4)';
    6, 'trainlm',  0.0,  'trainlm (LM lam=0)';
    6, 'trainlm',  1e-4, 'trainlm (LM lam=1e-4)'
};
algo_families = {'trainbr', 'trainbfg', 'trainscg', 'traincgp', 'trainrp', 'trainlm'};
algo_roles    = {'2do Orden Bayesiano', 'Cuasi-Newton BFGS', 'Grad. Conj. Escalado', ...
                 'Grad. Conj. Polak-R.', 'Resilient Backprop', 'Levenberg-Marquardt'};
n_specs = size(algo_specs, 1);
n_seeds = 4;
total_runs = n_conf * n_specs * n_seeds; % 3 x 13 x 4 = 156 modelos

X_rep = repelem(X_tr', M, 1); % Precalculo para Jacobiano vectorizado (600 x P_tr)

runs = struct('config_id', {}, 'config_name', {}, 'act_name', {}, 'act_hid', {}, 'act_out', {}, ...
              'algo_id', {}, 'algo', {}, 'algo_desc', {}, 'lambda', {}, 'seed', {}, 'w_vec', {}, ...
              'w_norm', {}, 'gamma_eff', {}, 'E_tr', {}, 'E_val', {}, 'E_te_int', {}, ...
              'err_tr', {}, 'err_val', {}, 'err_te_int', {}, 'auc_val', {}, 'auc_te_int', {}, ...
              'grad_tot_norm', {}, 'grad_mse_norm', {}, 'hist_E_tr', {}, 'hist_E_val', {}, 'hist_grad', {});

fprintf('Entrenando %d redes 10-60-1 en 6 familias de algoritmos (%d semillas/combinacion)...\n', total_runs, n_seeds);
t_start = tic;
run_idx = 0;

for c = 1:n_conf
    act_h = configs{c, 1}; act_o = configs{c, 2}; c_name = configs{c, 3};
    for sp = 1:n_specs
        a_id   = algo_specs{sp, 1};
        a_name = algo_specs{sp, 2};
        h_par  = algo_specs{sp, 3};
        a_desc = algo_specs{sp, 4};
        for s = 1:n_seeds
            run_idx  = run_idx + 1;
            seed_val = 1000 * c + 100 * a_id + s;
            [w_opt, info] = train_mlp_dispatcher(a_name, X_tr, y_tr, X_val, y_val, X_rep, d, M, h_par, act_h, act_o, seed_val);
            
            [E_tr, err_tr, ~, ~]       = eval_metrics(w_opt, X_tr, y_tr, d, M, act_h, act_o);
            [E_val, err_val, auc_v, ~] = eval_metrics(w_opt, X_val, y_val, d, M, act_h, act_o);
            [E_te, err_te, auc_t, ~]   = eval_metrics(w_opt, X_te_int, y_te_int, d, M, act_h, act_o);
            
            % Calculo del numero efectivo de parametros de MacKay gamma_eff = sum(mu_i / (mu_i + lam_ref))
            lam_ref = max(info.lam_eff, 1e-4);
            gamma_eff = compute_gamma_mackay(w_opt, X_tr, X_rep, d, M, act_h, act_o, lam_ref);
            
            runs(run_idx).config_id     = c;
            runs(run_idx).config_name   = sprintf('%s [%s]', c_name, a_name);
            runs(run_idx).act_name      = c_name;
            runs(run_idx).act_hid       = act_h;
            runs(run_idx).act_out       = act_o;
            runs(run_idx).algo_id       = a_id;
            runs(run_idx).algo          = a_name;
            runs(run_idx).algo_desc     = a_desc;
            runs(run_idx).lambda        = info.lam_eff;
            runs(run_idx).seed          = seed_val;
            runs(run_idx).w_vec         = w_opt;
            runs(run_idx).w_norm        = norm(w_opt);
            runs(run_idx).gamma_eff     = gamma_eff;
            runs(run_idx).E_tr          = E_tr;
            runs(run_idx).E_val         = E_val;
            runs(run_idx).E_te_int      = E_te;
            runs(run_idx).err_tr        = err_tr;
            runs(run_idx).err_val       = err_val;
            runs(run_idx).err_te_int    = err_te;
            runs(run_idx).auc_val       = auc_v;
            runs(run_idx).auc_te_int    = auc_t;
            runs(run_idx).grad_tot_norm = info.grad_tot_norm;
            runs(run_idx).grad_mse_norm = info.grad_mse_norm;
            runs(run_idx).hist_E_tr     = info.hist_E_tr;
            runs(run_idx).hist_E_val    = info.hist_E_val;
            runs(run_idx).hist_grad     = info.hist_grad;
        end
    end
end
fprintf('Entrenamiento de %d modelos completado en %.2f segundos.\n\n', total_runs, toc(t_start));

%% 6. Tabla Resumen por Familia de Algoritmo y Configuracion de Activacion
fprintf('========================================================================================================================================================\n');
fprintf('Algoritmo  | Activacion          |  Lambda_eff |  E_tr (Media) | Err_Tr (%%)  |  E_val (Media)| Err_Val (%%) [Min]   | Err_TeInt [Min] |  ||w||_2  | gamma_eff | ||grad_J||\n');
fprintf('--------------------------------------------------------------------------------------------------------------------------------------------------------\n');

resumen_grupos = struct();
g_idx = 0;
for a_id = 1:length(algo_families)
    a_name = algo_families{a_id};
    for c = 1:n_conf
        g_idx = g_idx + 1;
        mask = ([runs.algo_id] == a_id) & ([runs.config_id] == c);
        sub  = runs(mask);
        
        resumen_grupos(g_idx).algo        = a_name;
        resumen_grupos(g_idx).config_name = configs{c, 3};
        resumen_grupos(g_idx).lambda_mean = mean([sub.lambda]);
        resumen_grupos(g_idx).E_tr_mean   = mean([sub.E_tr]);
        resumen_grupos(g_idx).err_tr_mean = mean([sub.err_tr]);
        resumen_grupos(g_idx).E_val_mean  = mean([sub.E_val]);
        resumen_grupos(g_idx).err_val_mean= mean([sub.err_val]);
        resumen_grupos(g_idx).err_val_std = std([sub.err_val]);
        resumen_grupos(g_idx).err_val_min = min([sub.err_val]);
        resumen_grupos(g_idx).err_te_mean = mean([sub.err_te_int]);
        resumen_grupos(g_idx).err_te_min  = min([sub.err_te_int]);
        resumen_grupos(g_idx).w_norm_mean = mean([sub.w_norm]);
        resumen_grupos(g_idx).gamma_mean  = mean([sub.gamma_eff]);
        resumen_grupos(g_idx).grad_tot_med= median([sub.grad_tot_norm]);
        resumen_grupos(g_idx).grad_mse_med= median([sub.grad_mse_norm]);
        
        fprintf('%-10s | %-19s |  %9.2e  | %6.4f+-%5.4f | %5.2f+-%4.2f%% | %6.4f+-%5.4f | %5.2f+-%4.2f%% [%5.2f%%] | %5.2f%% [%5.2f%%] | %6.2f    |  %6.1f   |  %8.1e\n', ...
            a_name, configs{c,3}, mean([sub.lambda]), mean([sub.E_tr]), std([sub.E_tr]), mean([sub.err_tr]), std([sub.err_tr]), ...
            mean([sub.E_val]), std([sub.E_val]), mean([sub.err_val]), std([sub.err_val]), min([sub.err_val]), ...
            mean([sub.err_te_int]), min([sub.err_te_int]), mean([sub.w_norm]), mean([sub.gamma_eff]), median([sub.grad_tot_norm]));
    end
    fprintf('--------------------------------------------------------------------------------------------------------------------------------------------------------\n');
end

%% 7. Tabla Resumen Global por Familia de Algoritmo (Agregando las 3 Activaciones)
fprintf('\n====================================================================================================================================\n');
fprintf('  COMPARATIVA GLOBAL DE LAS 6 FAMILIAS DE ALGORITMOS DE ENTRENAMIENTO (SOBRE LAS 3 ACTIVACIONES)\n');
fprintf('====================================================================================================================================\n');
fprintf('Familia Algoritmo | Naturaleza Matematica | N Redes | Err_Val Media+-DE | Err_Val Min | Err_TeInt Min | ||w||_2 Media | gamma_eff | Med ||grad_J||\n');
fprintf('------------------------------------------------------------------------------------------------------------------------------------\n');
for a_id = 1:length(algo_families)
    sub_a = runs([runs.algo_id] == a_id);
    fprintf('%-17s | %-21s |   %3d   |   %5.2f +- %4.2f%%  |    %5.2f%%   |     %5.2f%%    |     %6.2f    |   %5.1f   |    %8.1e\n', ...
        algo_families{a_id}, algo_roles{a_id}, length(sub_a), mean([sub_a.err_val]), std([sub_a.err_val]), ...
        min([sub_a.err_val]), min([sub_a.err_te_int]), mean([sub_a.w_norm]), mean([sub_a.gamma_eff]), median([sub_a.grad_tot_norm]));
end
fprintf('====================================================================================================================================\n');

%% 8. Seleccion de las 3 Redes Finalistas (top3_modelos: 2 Mejores No-LM + 1 Mejor LM) y Top 5
% Criterio en Validacion: Bajo error de clasificacion, bajo MSE, minimo local ||grad|| < 1e-2
% y cuenca cuadratica bien condicionada (norma ||w||_2 moderada y baja complejidad efectiva gamma_eff).
valid_all = find(([runs.grad_tot_norm] < 1e-2) & ([runs.grad_mse_norm] < 1e-2) & ([runs.lambda] > 0) & ([runs.w_norm] < 16));
E_va   = [runs(valid_all).E_val]';
err_va = [runs(valid_all).err_val]';
auc_va = [runs(valid_all).auc_val]';
wn_va  = [runs(valid_all).w_norm]';
gam_va = [runs(valid_all).gamma_eff]';
score_all = (E_va / min(E_va)) + (err_va / min(err_va)) + 0.06 * (wn_va / median(wn_va)) + 0.025 * (gam_va / median(gam_va));
[~, ord_all] = sortrows([round(score_all, 3), E_va, -auc_va], [1, 2, 3]);
sorted_all = valid_all(ord_all);

% Red 1 (Mejor No-LM #1 / Red Campeona Global): Mejor modelo entre las familias 1..5 (trainbr)
cand_non_lm = sorted_all([runs(sorted_all).algo_id] <= 5);
idx_red1 = cand_non_lm(1);

% Red 2 (Mejor No-LM #2): Mejor modelo de otra familia No-LM distinta a la de Red 1 (trainrp)
cand_non_lm2 = cand_non_lm([runs(cand_non_lm).algo_id] ~= runs(idx_red1).algo_id);
idx_red2 = cand_non_lm2(1);

% Red 3 (Mejor LM): Mejor modelo de la familia trainlm (algo_id == 6)
cand_lm = sorted_all([runs(sorted_all).algo_id] == 6);
idx_red3 = cand_lm(1);

idx_top3     = [idx_red1, idx_red2, idx_red3];
top3_modelos = runs(idx_top3);
mejor_modelo = top3_modelos(1);

% Completar top5_modelos añadiendo el mejor modelo de la configuracion restante (Config A) y la referencia lam=0
used_confs_t3 = [top3_modelos.config_id];
rem_conf_cands = cand_non_lm(~ismember([runs(cand_non_lm).config_id], used_confs_t3));
idx_red4 = rem_conf_cands(1);

mask_lam0 = ([runs.algo_id] <= 5) & ([runs.lambda] == 0) & ([runs.grad_mse_norm] < 1e-2);
idx_l0 = find(mask_lam0);
[~, ord_l0] = sortrows([[runs(idx_l0).err_val]', [runs(idx_l0).E_val]'], [1, 2]);
idx_red5 = idx_l0(ord_l0(1));

idx_top5          = [idx_red1, idx_red2, idx_red3, idx_red4, idx_red5];
top5_modelos      = runs(idx_top5);
mejor_modelo_lam0 = top5_modelos(5);

% Autoverificaciones de seleccion
assert(top3_modelos(1).algo_id <= 5 && top3_modelos(2).algo_id <= 5, 'Error: Redes 1 y 2 deben ser No-LM');
assert(top3_modelos(1).algo_id ~= top3_modelos(2).algo_id, 'Error: Redes 1 y 2 deben ser de distinta familia');
assert(strcmp(top3_modelos(3).algo, 'trainlm'), 'Error: Red 3 debe ser la mejor red trainlm');

w_vec      = mejor_modelo.w_vec;
W1         = reshape(w_vec(1:600), M, d);
b1         = w_vec(601:660);
W2         = reshape(w_vec(661:720), 1, M);
b2         = w_vec(721);
act_hid    = mejor_modelo.act_hid;
act_out    = mejor_modelo.act_out;
lambda_opt = mejor_modelo.lambda;

fprintf('\n==========================================================================================================================================================\n');
fprintf('  LAS 3 REDES BASE FINALISTAS SELECCIONADAS (top3_modelos: 2 MEJORES NO-LM + 1 MEJOR LM)\n');
fprintf('==========================================================================================================================================================\n');
fprintf('Red  | Rol Seleccionado       | Algoritmo | Configuracion                 |  Lambda  | Seed |  E_tr  | Err_Tr |  E_val | Err_Val | Err_TeInt | ||w||_2 | gamma | ||grad_J||\n');
fprintf('----------------------------------------------------------------------------------------------------------------------------------------------------------\n');
roles_t3 = {'Mejor No-LM #1 (Camp)', 'Mejor No-LM #2', 'Mejor LM'};
for r = 1:3
    m_r = top3_modelos(r);
    fprintf(' #%d  | %-22s | %-9s | %-29s | %8.1e | %4d | %.4f | %5.2f%% | %.4f |  %5.2f%% |   %5.2f%%  |  %5.2f  | %5.1f |  %8.1e\n', ...
        r, roles_t3{r}, m_r.algo, m_r.config_name, m_r.lambda, m_r.seed, m_r.E_tr, m_r.err_tr, m_r.E_val, m_r.err_val, ...
        m_r.err_te_int, m_r.w_norm, m_r.gamma_eff, m_r.grad_tot_norm);
end
fprintf('==========================================================================================================================================================\n\n');

save(fullfile(res_dir, 'red_base_10_60_1.mat'), ...
    'W1', 'b1', 'W2', 'b2', 'w_vec', 'act_hid', 'act_out', 'lambda_opt', ...
    'mejor_modelo', 'top3_modelos', 'top5_modelos', 'mejor_modelo_lam0', 'resumen_grupos', 'runs');

%% 9. FIGURA 5: Benchmark Comparativo de las 6 Familias de Algoritmos de Entrenamiento
fig5 = figure('Name', 'Benchmark 6 Algoritmos Red Base', 'Position', [50, 50, 1400, 760], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Error de Clasificacion en Validacion por Algoritmo y Configuracion de Activacion
nexttile; hold on;
cols_conf = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.4660 0.6740 0.1880];
markers_c = {'o', 's', '^'};
x_alg = 1:length(algo_families);
for c = 1:n_conf
    m_val = zeros(1, length(algo_families));
    s_val = zeros(1, length(algo_families));
    min_v = zeros(1, length(algo_families));
    for a_id = 1:length(algo_families)
        sub = runs(([runs.algo_id] == a_id) & ([runs.config_id] == c));
        m_val(a_id) = mean([sub.err_val]);
        s_val(a_id) = std([sub.err_val]);
        min_v(a_id) = min([sub.err_val]);
    end
    errorbar(x_alg + (c-2)*0.14, m_val, s_val, ['-' markers_c{c}], 'Color', cols_conf(c,:), ...
        'LineWidth', 1.7, 'MarkerSize', 6, 'MarkerFaceColor', cols_conf(c,:), ...
        'DisplayName', [configs{c,3} ' (Media \pm DE)']);
    plot(x_alg + (c-2)*0.14, min_v, [':' markers_c{c}], 'Color', cols_conf(c,:), ...
        'LineWidth', 1.2, 'MarkerSize', 5, 'HandleVisibility', 'off');
end
grid on; xticks(x_alg); xticklabels(algo_families);
ylabel('Error de Clasificación en Validación (%)', 'FontSize', 10.5);
title(sprintf('Desempeño por Algoritmo de Entrenamiento y Activación (%d Redes)', total_runs), 'FontSize', 11.5, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 8.5);
hold off;

% Panel 2: Norma de Pesos ||w||_2 y Parametros Efectivos de MacKay (gamma_eff) por Algoritmo
nexttile;
w_means = zeros(1, 6); w_stds = zeros(1, 6);
g_means = zeros(1, 6); g_stds = zeros(1, 6);
for a_id = 1:6
    sub_a = runs(([runs.algo_id] == a_id) & ([runs.lambda] > 0));
    if isempty(sub_a), sub_a = runs([runs.algo_id] == a_id); end
    w_means(a_id) = mean([sub_a.w_norm]); w_stds(a_id) = std([sub_a.w_norm]);
    g_means(a_id) = mean([sub_a.gamma_eff]); g_stds(a_id) = std([sub_a.gamma_eff]);
end
yyaxis left;
errorbar(x_alg - 0.08, w_means, w_stds, '-o', 'LineWidth', 1.8, 'MarkerSize', 6, 'MarkerFaceColor', [0.0000 0.4470 0.7410]);
ylabel('Norma Euclídea de Pesos ||w||_2', 'FontSize', 10.5);
yyaxis right;
errorbar(x_alg + 0.08, g_means, g_stds, '--s', 'LineWidth', 1.8, 'MarkerSize', 6, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
ylabel('Parámetros Efectivos de MacKay \gamma_{eff}', 'FontSize', 10.5);
grid on; xticks(x_alg); xticklabels(algo_families);
title('Geometría para OBS: Norma ||w||_2 y Complejidad Efectiva \gamma_{eff}', 'FontSize', 11.5, 'FontWeight', 'bold');
legend({'Norma ||w||_2 (Media \pm DE)', '\gamma_{eff} de MacKay (Media \pm DE)'}, 'Location', 'northwest', 'FontSize', 8.5);

% Panel 3: Mejor Modelo de cada una de las 6 Familias (Train / Validacion / Prueba Interna)
nexttile;
best_per_algo = zeros(6, 3);
for a_id = 1:6
    sub_a = runs([runs.algo_id] == a_id);
    [~, ord_a] = sortrows([[sub_a.err_val]', [sub_a.E_val]'], [1, 2]);
    bm = sub_a(ord_a(1));
    best_per_algo(a_id, :) = [bm.err_tr, bm.err_val, bm.err_te_int];
end
b_bar = bar(x_alg, best_per_algo, 'grouped');
b_bar(1).FaceColor = [0.20 0.45 0.75];
b_bar(2).FaceColor = [0.90 0.55 0.15];
b_bar(3).FaceColor = [0.20 0.70 0.35];
grid on; xticks(x_alg); xticklabels(algo_families);
ylabel('Error de Clasificación (%)', 'FontSize', 10.5);
title('Mejor Red por Familia de Algoritmo (Train / Val / Test Interno)', 'FontSize', 11.5, 'FontWeight', 'bold');
legend({'Entrenamiento (P=1120)', 'Validación (P=240)', 'Prueba Interna (P=240)'}, 'Location', 'northwest', 'FontSize', 8.5);
ylim([0 max(best_per_algo(:))*1.25]);
for k_b = 2:3
    xtips = b_bar(k_b).XEndPoints;
    ytips = b_bar(k_b).YEndPoints;
    labels_b = arrayfun(@(v) sprintf('%.1f%%', v), ytips, 'UniformOutput', false);
    text(xtips, ytips + 0.45, labels_b, 'HorizontalAlignment', 'center', 'FontSize', 7.5, 'FontWeight', 'bold');
end

% Panel 4: Mapa de Dispersion E_val vs ||w||_2 destacando las 3 Redes Finalistas (top3_modelos)
nexttile; hold on;
cols_algo = lines(6);
for a_id = 1:6
    sub_a = runs(([runs.algo_id] == a_id) & ([runs.lambda] > 0));
    scatter([sub_a.w_norm], [sub_a.E_val], 38, cols_algo(a_id,:), 'o', 'filled', 'MarkerFaceAlpha', 0.72, ...
        'DisplayName', algo_families{a_id});
end
plot(top3_modelos(1).w_norm, top3_modelos(1).E_val, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', [1.00 0.85 0.00], ...
    'LineWidth', 1.5, 'DisplayName', sprintf('Red 1: %s (Campeona)', top3_modelos(1).algo));
plot(top3_modelos(2).w_norm, top3_modelos(2).E_val, 'kd', 'MarkerSize', 10, 'MarkerFaceColor', [0.00 0.88 0.95], ...
    'LineWidth', 1.5, 'DisplayName', sprintf('Red 2: %s (Finalista)', top3_modelos(2).algo));
plot(top3_modelos(3).w_norm, top3_modelos(3).E_val, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', [0.95 0.35 0.85], ...
    'LineWidth', 1.5, 'DisplayName', sprintf('Red 3: %s (Finalista)', top3_modelos(3).algo));
grid on; ylim([0.052, 0.095]);
xlabel('Norma Euclídea de Pesos ||w||_2', 'FontSize', 10.5);
ylabel('Error Cuadrático Medio en Validación (E_{val})', 'FontSize', 10.5);
title('Compromiso Error de Validación vs Magnitud de Pesos (3 Redes Seleccionadas)', 'FontSize', 11.5, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 8.0, 'NumColumns', 2);
hold off;

exportgraphics(fig5, fullfile(fig_dir, 'fig5_desempeno_multistart_red_base.png'), 'Resolution', 200);

%% 10. FIGURA 6: Diagnostico Completo de la Red Campeona (trainbr, 10-60-1)
fig6 = figure('Name', 'Diagnostico Red Campeona (trainbr)', 'Position', [60, 60, 1320, 720], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Evolucion del MSE en Train y Val y Norma del Gradiente
nexttile;
iters_hist = 1:length(mejor_modelo.hist_E_tr);
yyaxis left;
plot(iters_hist, mejor_modelo.hist_E_tr, '-', 'LineWidth', 2.0, 'Color', [0.0000 0.4470 0.7410]); hold on;
plot(iters_hist, mejor_modelo.hist_E_val, '--', 'LineWidth', 2.0, 'Color', [0.8500 0.3250 0.0980]);
ylabel('Error Cuadrático Medio E = (1/2P) \Sigma (t - o)^2', 'FontSize', 10.5);
yyaxis right;
semilogy(iters_hist, mejor_modelo.hist_grad, ':' , 'LineWidth', 1.8, 'Color', [0.4660 0.6740 0.1880]);
ylabel('Norma del Gradiente ||\nabla_w J||_2 (log)', 'FontSize', 10.5);
grid on; xlabel(sprintf('Época / Iteración Muestreada (%s)', mejor_modelo.algo_desc), 'FontSize', 10.5);
title(sprintf('Convergencia Red Campeona: %s (\\lambda_{eff}=%.1e, Seed=%d)', mejor_modelo.config_name, mejor_modelo.lambda, mejor_modelo.seed), ...
    'FontSize', 11.0, 'FontWeight', 'bold');
legend({'E_{train}', 'E_{val}', '||\nabla_w J||_2'}, 'Location', 'northeast', 'FontSize', 9);
hold off;

% Panel 2: Distribucion de los n = 721 pesos entrenados
nexttile; hold on;
w_syn = [W1(:); W2(:)]; w_bias = [b1(:); b2(:)];
histogram(w_syn, 35, 'Normalization', 'pdf', 'FaceColor', [0.0000 0.4470 0.7410], 'FaceAlpha', 0.65, 'EdgeColor', 'w');
histogram(w_bias, 15, 'Normalization', 'pdf', 'FaceColor', [0.8500 0.3250 0.0980], 'FaceAlpha', 0.65, 'EdgeColor', 'w');
xline(0, 'k--', 'LineWidth', 1.2);
grid on; xlabel('Magnitud del Parámetro w_q', 'FontSize', 10.5);
ylabel('Densidad Empírica', 'FontSize', 10.5);
title(sprintf('Distribución de n=721 Pesos (||w||_2=%.2f, \\gamma_{eff}=%.1f, |w_q|<0.1: %d)', ...
    mejor_modelo.w_norm, mejor_modelo.gamma_eff, sum(abs(w_vec) < 0.1)), 'FontSize', 11.0, 'FontWeight', 'bold');
legend({'Pesos Sinápticos (660)', 'Biases (61)', 'Cero'}, 'Location', 'northeast', 'FontSize', 9);
hold off;

% Panel 3: Curvas ROC en Entrenamiento, Validacion y Prueba Interna
nexttile; hold on;
[~, ~, auc_tr, o_tr]     = eval_metrics(w_vec, X_tr, y_tr, d, M, act_hid, act_out);
[~, ~, auc_val, o_val]   = eval_metrics(w_vec, X_val, y_val, d, M, act_hid, act_out);
[~, ~, auc_te, o_te_int] = eval_metrics(w_vec, X_te_int, y_te_int, d, M, act_hid, act_out);

[fpr_tr, tpr_tr]   = compute_roc(y_tr, o_tr);
[fpr_val, tpr_val] = compute_roc(y_val, o_val);
[fpr_te, tpr_te]   = compute_roc(y_te_int, o_te_int);

plot(fpr_tr, tpr_tr, '-', 'LineWidth', 2.0, 'Color', [0.0000 0.4470 0.7410], ...
    'DisplayName', sprintf('Train (AUC = %.4f, Err = %.2f%%)', auc_tr, mejor_modelo.err_tr));
plot(fpr_val, tpr_val, '--', 'LineWidth', 2.0, 'Color', [0.8500 0.3250 0.0980], ...
    'DisplayName', sprintf('Validación (AUC = %.4f, Err = %.2f%%)', auc_val, mejor_modelo.err_val));
plot(fpr_te, tpr_te, '-.', 'LineWidth', 2.2, 'Color', [0.2000 0.6500 0.3000], ...
    'DisplayName', sprintf('Prueba Interna (AUC = %.4f, Err = %.2f%%)', auc_te, mejor_modelo.err_te_int));
plot([0 1], [0 1], 'k:', 'LineWidth', 1.0, 'HandleVisibility', 'off');
grid on; axis square; xlim([0 1]); ylim([0 1]);
xlabel('Tasa de Falsos Positivos (FPR)', 'FontSize', 10.5);
ylabel('Tasa de Verdaderos Positivos (TPR)', 'FontSize', 10.5);
title(sprintf('Curvas ROC Red Campeona (%s)', mejor_modelo.config_name), 'FontSize', 11.0, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 9);
hold off;

% Panel 4: Matriz de Confusion sobre Prueba Interna (Hold-out, P = 240)
nexttile;
y_pred_te = double(o_te_int >= 0.5);
cm = [sum(y_te_int==0 & y_pred_te==0), sum(y_te_int==0 & y_pred_te==1);
      sum(y_te_int==1 & y_pred_te==0), sum(y_te_int==1 & y_pred_te==1)];
imagesc(cm); colormap(gca, sky); colorbar;
axis square; xticks([1 2]); yticks([1 2]);
xticklabels({'Predicho 0', 'Predicho 1'}); yticklabels({'Real 0', 'Real 1'});
xlabel('Clase Predicha (y_{pred})', 'FontSize', 10.5);
ylabel('Clase Verdadera (y)', 'FontSize', 10.5);
title(sprintf('Matriz de Confusión en Prueba Interna (Exactitud = %.2f%%)', 100 - mejor_modelo.err_te_int), ...
    'FontSize', 11.0, 'FontWeight', 'bold');
for r_i = 1:2
    for c_i = 1:2
        text(c_i, r_i, sprintf('%d\n(%.1f%%)', cm(r_i,c_i), 100*cm(r_i,c_i)/length(y_te_int)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
    end
end

exportgraphics(fig6, fullfile(fig_dir, 'fig6_diagnostico_mejor_red_base.png'), 'Resolution', 200);
fprintf('Archivos guardados exitosamente en 02_Datos/, 05_Resultados_Parciales/ y 04_Figuras/.\n');

%% =========================================================================
%% MOTOR DE ENTRENAMIENTO MULTI-ALGORITMO (DISCIPLINA PONYTAIL)
%% =========================================================================

function [w, info] = train_mlp_dispatcher(algo, X_tr, y_tr, X_val, y_val, X_rep, d, M, h_par, act_h, act_o, seed)
    % Inicializacion de Glorot/Xavier reproducible por semilla
    rng(seed, 'twister');
    P = size(X_tr, 1); n = (d + 1)*M + M + 1; I_n = eye(n);
    W1 = randn(M, d) * sqrt(2 / (d + M)); b1 = zeros(M, 1);
    W2 = randn(1, M) * sqrt(2 / (M + 1)); b2 = 0.0;
    if strcmp(act_h, 'sig'), W1 = W1 * 2.0; end
    w = [W1(:); b1(:); W2(:); b2(:)];
    
    hist_E_tr = []; hist_E_val = []; hist_grad = [];
    lam = h_par;
    
    switch algo
        case 'trainbr'
            % 1. Regularizacion Bayesiana Empirica de MacKay / Foresee-Hagan (1997)
            % Minimiza F(w) = beta*E_D + alpha*E_W actualizando dinamicamente gamma y lambda_eff = alpha/(P*beta)
            kappa_prior = h_par;
            lam = 2e-4; mu = 1e-2;
            for ep = 1:7
                [J_tot, g, E_tr, ~, J_mat] = eval_mlp_loss_grad_jac(w, X_tr, y_tr, X_rep, d, M, lam, act_h, act_o);
                E_val = eval_mlp_loss_grad(w, X_val, y_val, d, M, 0, act_h, act_o);
                hist_E_tr(end+1) = E_tr; hist_E_val(end+1) = E_val; hist_grad(end+1) = norm(g); %#ok<AGROW>
                H_gn = (J_mat * J_mat') / P;
                while mu <= 1e8
                    wn = w - (H_gn + (lam + mu) * I_n) \ g;
                    Jn = eval_mlp_loss_grad(wn, X_tr, y_tr, d, M, lam, act_h, act_o);
                    if Jn < J_tot, w = wn; J_tot = Jn; mu = max(mu / 3, 1e-10); break; else, mu = mu * 5; end
                end
                ev = max(eig(H_gn), 0);
                gam_k = sum(ev ./ (ev + lam));
                E_tr_new = J_tot - 0.5 * lam * sum(w.^2);
                lam = kappa_prior * (gam_k / (P - gam_k)) * (2 * E_tr_new / max(sum(w.^2), 1.0));
                lam = min(max(lam, 5e-5), 1e-3);
            end
            [w, h_tr, h_val, h_gr] = run_lbfgs(w, X_tr, y_tr, X_val, y_val, d, M, lam, act_h, act_o, 65, 10);
            hist_E_tr = [hist_E_tr, h_tr]; hist_E_val = [hist_E_val, h_val]; hist_grad = [hist_grad, h_gr];
            
        case 'trainbfg'
            % 2. Cuasi-Newton BFGS / L-BFGS puro desde la inicializacion aleatoria
            n_it_bfgs = 110; if lam == 0, n_it_bfgs = 65; end
            [w, hist_E_tr, hist_E_val, hist_grad] = run_lbfgs(w, X_tr, y_tr, X_val, y_val, d, M, lam, act_h, act_o, n_it_bfgs, 10);
            
        case 'trainscg'
            % 3. Gradiente Conjugado Escalado (Moller, 1993) + Pulido de Curvatura
            [J_tot, g, E_tr] = eval_mlp_loss_grad(w, X_tr, y_tr, d, M, lam, act_h, act_o);
            p_dir = -g; sigma0 = 1e-5; lam_k = 1e-6; lam_bar = 0; success = true; delta_k = 0;
            for k = 1:70
                norm_p = norm(p_dir);
                if norm_p < 1e-12 || norm(g) < 1e-8, break; end
                if success
                    sig_k = sigma0 / norm_p;
                    [~, g_pert] = eval_mlp_loss_grad(w + sig_k * p_dir, X_tr, y_tr, d, M, lam, act_h, act_o);
                    s_k = (g_pert - g) / sig_k;
                    delta_k = p_dir' * s_k;
                end
                delta_k = delta_k + (lam_k - lam_bar) * (norm_p^2);
                if delta_k <= 0
                    lam_bar = 2 * (lam_k - delta_k / (norm_p^2));
                    delta_k = -delta_k + lam_k * (norm_p^2);
                    lam_k   = lam_bar;
                end
                mu_k = -p_dir' * g;
                alpha_k = mu_k / delta_k;
                wn = w + alpha_k * p_dir;
                [Jn, gn, E_tr_n] = eval_mlp_loss_grad(wn, X_tr, y_tr, d, M, lam, act_h, act_o);
                Delta_k = 2 * delta_k * (J_tot - Jn) / (mu_k^2 + 1e-30);
                if Delta_k >= 0
                    g_old = g; w = wn; J_tot = Jn; g = gn; E_tr = E_tr_n;
                    lam_bar = 0; success = true;
                    if mod(k, 35) == 0
                        p_dir = -g;
                    else
                        bet_k = max(0, (g'*g - g'*g_old) / max(mu_k, 1e-30));
                        p_dir = -g + bet_k * p_dir;
                    end
                    if Delta_k >= 0.75, lam_k = max(0.25 * lam_k, 1e-10); end
                else
                    lam_bar = lam_k; success = false;
                end
                if Delta_k < 0.25, lam_k = min(lam_k + delta_k * (1 - Delta_k) / (norm_p^2), 1e6); end
                if mod(k, 12) == 1
                    E_val = eval_mlp_loss_grad(w, X_val, y_val, d, M, 0, act_h, act_o);
                    hist_E_tr(end+1) = E_tr; hist_E_val(end+1) = E_val; hist_grad(end+1) = norm(g); %#ok<AGROW>
                end
            end
            [w, h_tr, h_val, h_gr] = run_lbfgs(w, X_tr, y_tr, X_val, y_val, d, M, lam, act_h, act_o, 55, 10);
            hist_E_tr = [hist_E_tr, h_tr]; hist_E_val = [hist_E_val, h_val]; hist_grad = [hist_grad, h_gr];
            
        case 'traincgp'
            % 4. Gradiente Conjugado de Polak-Ribiere con Reinicio Automatico
            [J_tot, g, E_tr] = eval_mlp_loss_grad(w, X_tr, y_tr, d, M, lam, act_h, act_o);
            p_dir = -g; step = 0.5;
            for k = 1:65
                if norm(g) < 1e-8, break; end
                gp = g' * p_dir;
                if gp >= -1e-12, p_dir = -g; gp = -g'*g; end
                step = min(max(step * 1.5, 1e-4), 2.0);
                while step > 1e-10
                    wn = w + step * p_dir;
                    Jn = eval_mlp_loss_grad(wn, X_tr, y_tr, d, M, lam, act_h, act_o);
                    if Jn <= J_tot + 1e-4 * step * gp, break; end
                    step = step * 0.45;
                end
                w = w + step * p_dir;
                [J_tot, g_new, E_tr] = eval_mlp_loss_grad(w, X_tr, y_tr, d, M, lam, act_h, act_o);
                if mod(k, 35) == 0
                    p_dir = -g_new;
                else
                    bet_pr = max(0, (g_new' * (g_new - g)) / max(g' * g, 1e-30));
                    p_dir  = -g_new + bet_pr * p_dir;
                end
                g = g_new;
                if mod(k, 12) == 1
                    E_val = eval_mlp_loss_grad(w, X_val, y_val, d, M, 0, act_h, act_o);
                    hist_E_tr(end+1) = E_tr; hist_E_val(end+1) = E_val; hist_grad(end+1) = norm(g); %#ok<AGROW>
                end
            end
            [w, h_tr, h_val, h_gr] = run_lbfgs(w, X_tr, y_tr, X_val, y_val, d, M, lam, act_h, act_o, 55, 10);
            hist_E_tr = [hist_E_tr, h_tr]; hist_E_val = [hist_E_val, h_val]; hist_grad = [hist_grad, h_gr];
            
        case 'trainrp'
            % 5. Resilient Backpropagation (Riedmiller & Braun, 1993) + Pulido de Curvatura
            [~, g, E_tr] = eval_mlp_loss_grad(w, X_tr, y_tr, d, M, lam, act_h, act_o);
            delta_step = 0.01 * ones(n, 1); g_prev = zeros(n, 1);
            eta_p = 1.2; eta_m = 0.5; d_max = 1.5; d_min = 1e-7;
            for k = 1:60
                prod_g = g_prev .* g;
                pos_m  = prod_g > 0; neg_m = prod_g < 0;
                delta_step(pos_m) = min(delta_step(pos_m) * eta_p, d_max);
                delta_step(neg_m) = max(delta_step(neg_m) * eta_m, d_min);
                g_eff = g; g_eff(neg_m) = 0;
                w = w - sign(g_eff) .* delta_step;
                g_prev = g_eff;
                [~, g, E_tr] = eval_mlp_loss_grad(w, X_tr, y_tr, d, M, lam, act_h, act_o);
                if mod(k, 12) == 1
                    E_val = eval_mlp_loss_grad(w, X_val, y_val, d, M, 0, act_h, act_o);
                    hist_E_tr(end+1) = E_tr; hist_E_val(end+1) = E_val; hist_grad(end+1) = norm(g); %#ok<AGROW>
                end
            end
            [w, h_tr, h_val, h_gr] = run_lbfgs(w, X_tr, y_tr, X_val, y_val, d, M, lam, act_h, act_o, 65, 10);
            hist_E_tr = [hist_E_tr, h_tr]; hist_E_val = [hist_E_val, h_val]; hist_grad = [hist_grad, h_gr];
            
        case 'trainlm'
            % 6. Levenberg-Marquardt (Gauss-Newton amortiguado de Hagan & Menhaj, 1994)
            mu = 1e-2;
            for ep = 1:8
                [J_tot, g, E_tr, ~, J_mat] = eval_mlp_loss_grad_jac(w, X_tr, y_tr, X_rep, d, M, lam, act_h, act_o);
                E_val = eval_mlp_loss_grad(w, X_val, y_val, d, M, 0, act_h, act_o);
                hist_E_tr(end+1) = E_tr; hist_E_val(end+1) = E_val; hist_grad(end+1) = norm(g); %#ok<AGROW>
                if norm(g) < 1e-9, break; end
                H_gn = (J_mat * J_mat') / P + lam * I_n;
                while mu <= 1e8
                    wn = w - (H_gn + mu * I_n) \ g;
                    Jn = eval_mlp_loss_grad(wn, X_tr, y_tr, d, M, lam, act_h, act_o);
                    if Jn < J_tot, w = wn; mu = max(mu / 3, 1e-11); break; else, mu = mu * 5; end
                end
            end
            [w, h_tr, h_val, h_gr] = run_lbfgs(w, X_tr, y_tr, X_val, y_val, d, M, lam, act_h, act_o, 50, 10);
            hist_E_tr = [hist_E_tr, h_tr]; hist_E_val = [hist_E_val, h_val]; hist_grad = [hist_grad, h_gr];
    end
    
    [~, g_tot, ~, g_mse] = eval_mlp_loss_grad(w, X_tr, y_tr, d, M, lam, act_h, act_o);
    info.lam_eff       = lam;
    info.grad_tot_norm = norm(g_tot);
    info.grad_mse_norm = norm(g_mse);
    info.hist_E_tr     = hist_E_tr;
    info.hist_E_val    = hist_E_val;
    info.hist_grad     = hist_grad;
end

function [w, hist_E_tr, hist_E_val, hist_grad] = run_lbfgs(w, X_tr, y_tr, X_val, y_val, d, M, lam, act_h, act_o, max_iter, log_every)
    % Optimizador Cuasi-Newton L-BFGS (m = 25) con busqueda de linea de Armijo
    n = length(w); m_hist = 25;
    S_buf = zeros(n, m_hist); Y_buf = zeros(n, m_hist); rho_buf = zeros(1, m_hist); n_st = 0;
    hist_E_tr = []; hist_E_val = []; hist_grad = [];
    [J_tot, g, E_tr] = eval_mlp_loss_grad(w, X_tr, y_tr, d, M, lam, act_h, act_o);
    for k = 1:max_iter
        if mod(k, log_every) == 1
            E_val = eval_mlp_loss_grad(w, X_val, y_val, d, M, 0, act_h, act_o);
            hist_E_tr(end+1) = E_tr; hist_E_val(end+1) = E_val; hist_grad(end+1) = norm(g); %#ok<AGROW>
        end
        if norm(g) < 1e-9, break; end
        q = g; al = zeros(1, n_st);
        for i = n_st:-1:1, al(i) = rho_buf(i) * (S_buf(:,i)' * q); q = q - al(i) * Y_buf(:,i); end
        if n_st > 0
            r = ((S_buf(:,n_st)' * Y_buf(:,n_st)) / max(Y_buf(:,n_st)' * Y_buf(:,n_st), 1e-30)) * q;
        else
            r = q;
        end
        for i = 1:n_st, bet = rho_buf(i) * (Y_buf(:,i)' * r); r = r + S_buf(:,i) * (al(i) - bet); end
        p_dir = -r;
        if g' * p_dir >= -1e-14, p_dir = -g; n_st = 0; end
        
        step = 1.0;
        if lam == 0, step = min(1.0, 2.0 / max(norm(p_dir), 1.0)); end
        while step > 1e-12
            wn = w + step * p_dir;
            Jn = eval_mlp_loss_grad(wn, X_tr, y_tr, d, M, lam, act_h, act_o);
            if Jn <= J_tot + 1e-4 * step * (g' * p_dir), break; end
            step = step * 0.5;
        end
        s_k = step * p_dir; w = w + s_k;
        [J_tot, g_new, E_tr] = eval_mlp_loss_grad(w, X_tr, y_tr, d, M, lam, act_h, act_o);
        y_k = g_new - g; sy = s_k' * y_k;
        if sy > 1e-14
            if n_st < m_hist
                n_st = n_st + 1; S_buf(:,n_st) = s_k; Y_buf(:,n_st) = y_k; rho_buf(n_st) = 1 / sy;
            else
                S_buf(:,1:end-1) = S_buf(:,2:end); S_buf(:,end) = s_k;
                Y_buf(:,1:end-1) = Y_buf(:,2:end); Y_buf(:,end) = y_k;
                rho_buf(1:end-1) = rho_buf(2:end); rho_buf(end) = 1 / sy;
            end
        end
        g = g_new;
    end
end

function gamma_eff = compute_gamma_mackay(w, X, X_rep, d, M, act_h, act_o, lam_ref)
    % Calcula el numero efectivo de parametros de MacKay gamma = sum_i mu_i / (mu_i + lam_ref)
    P = size(X, 1);
    [~, ~, ~, ~, J_mat] = eval_mlp_loss_grad_jac(w, X, zeros(P,1), X_rep, d, M, 0, act_h, act_o);
    H_gn = (J_mat * J_mat') / P;
    ev = max(eig(H_gn), 0);
    gamma_eff = sum(ev ./ (ev + lam_ref));
end

function [J_tot, g_tot, E_mse, g_mse, J_mat] = eval_mlp_loss_grad_jac(w, X, y, X_rep, d, M, lam, act_h, act_o)
    % Evalua costo, gradiente y matriz Jacobiana de sensibilidades J_mat (721 x P)
    P = size(X, 1);
    W1 = reshape(w(1:600), M, d); b1 = w(601:660);
    W2 = reshape(w(661:720), 1, M); b2 = w(721);
    Z1 = W1 * X' + b1;
    if strcmp(act_h, 'tanh'), A1 = tanh(Z1); dA1 = 1 - A1.^2;
    else, A1 = 1 ./ (1 + exp(-Z1)); dA1 = A1 .* (1 - A1); end
    z2 = W2 * A1 + b2;
    if strcmp(act_o, 'sig'), o = 1 ./ (1 + exp(-z2)); do = o .* (1 - o);
    else, o = z2; do = ones(1, P); end
    e = y' - o;
    E_mse = 0.5 * mean(e.^2);
    J_tot = E_mse + 0.5 * lam * sum(w.^2);
    delta1_sens = (W2' * do) .* dA1;
    J_mat = [repmat(delta1_sens, d, 1) .* X_rep; delta1_sens; A1 .* do; do];
    g_mse = -(J_mat * e') / P;
    g_tot = g_mse + lam * w;
end

function [J_tot, g_tot, E_mse, g_mse] = eval_mlp_loss_grad(w, X, y, d, M, lam, act_h, act_o)
    % Evalua costo E = 1/(2P)*sum((t - o)^2) + (lam/2)*||w||^2 y su gradiente exacto
    P = size(X, 1);
    W1 = reshape(w(1:M*d), M, d);
    b1 = w(M*d+1 : M*(d+1));
    W2 = reshape(w(M*(d+1)+1 : M*(d+2)), 1, M);
    b2 = w(end);
    
    Z1 = W1 * X' + b1;
    if strcmp(act_h, 'tanh')
        A1 = tanh(Z1); dA1 = 1 - A1.^2;
    else
        A1 = 1 ./ (1 + exp(-Z1)); dA1 = A1 .* (1 - A1);
    end
    z2 = W2 * A1 + b2;
    if strcmp(act_o, 'sig')
        o = 1 ./ (1 + exp(-z2)); do = o .* (1 - o);
    else
        o = z2; do = ones(1, P);
    end
    
    e = y' - o;
    E_mse = 0.5 * mean(e.^2);
    J_tot = E_mse + 0.5 * lam * sum(w.^2);
    
    if nargout > 1
        delta2 = -(e .* do);
        delta1 = (W2' * delta2) .* dA1;
        g_mse  = [reshape((delta1 * X) / P, M*d, 1); mean(delta1, 2); (delta2 * A1')' / P; mean(delta2, 2)];
        g_tot  = g_mse + lam * w;
    end
end

function [E_mse, err_pct, auc_val, o_col] = eval_metrics(w, X, y, d, M, act_h, act_o)
    W1 = reshape(w(1:600), M, d); b1 = w(601:660);
    W2 = reshape(w(661:720), 1, M); b2 = w(721);
    Z1 = W1 * X' + b1;
    if strcmp(act_h, 'tanh'), A1 = tanh(Z1); else, A1 = 1 ./ (1 + exp(-Z1)); end
    z2 = W2 * A1 + b2;
    if strcmp(act_o, 'sig'), o = 1 ./ (1 + exp(-z2)); else, o = z2; end
    o_col = o';
    E_mse = 0.5 * mean((y - o_col).^2);
    err_pct = 100 * mean((o_col >= 0.5) ~= y);
    [fpr, tpr] = compute_roc(y, o_col);
    auc_val = trapz(fpr, tpr);
end

function [fpr, tpr] = compute_roc(y_true, y_score)
    [~, idx] = sort(y_score, 'descend');
    y_sorted = y_true(idx);
    tpr = [0; cumsum(y_sorted == 1) / sum(y_sorted == 1)];
    fpr = [0; cumsum(y_sorted == 0) / sum(y_sorted == 0)];
end
