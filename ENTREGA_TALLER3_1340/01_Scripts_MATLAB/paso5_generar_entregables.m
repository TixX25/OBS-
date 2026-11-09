%% PASO 5: SIMULACION COMPARATIVA EN X_TEST Y ENTREGABLES DE LAS 2 REDES CAMPEONAS (OBS PURO)
% Evalua y empaqueta de forma completa e independiente en carpetas separadas:
%   1) 06_Entregables/Candidata_1_trainbr/ (Red #1: trainbr | Config B: tanh+lin, 20 neuronas, 107 pesos)
%   2) 06_Entregables/Candidata_2_trainlm/ (Red #2: trainlm | Config C: sig+sig,  17 neuronas,  90 pesos)
% Entregables oficiales segun enunciado (TallerIII.pdf): OBS_1340.MAT y pesos_1340.mat,
% junto con copias de compatibilidad (OBS_13_40.mat, OBS_30.MAT, etc.) y sus 4 figuras.
% CERO REENTRENAMIENTO: Poda OBS 100%% analitica segun Ec. 5 de Hassibi & Stork.

clear; clc; close all;

%% 1. Carga de datos y resultados de los Puntos 2 y 4
base_dir  = fileparts(fileparts(mfilename('fullpath')));
if isempty(base_dir), base_dir = pwd; end
addpath(fullfile(base_dir, '03_Scripts'));
datos_dir = fullfile(base_dir, '02_Datos');
fig_dir   = fullfile(base_dir, '04_Figuras');
res_dir   = fullfile(base_dir, '05_Resultados_Parciales');
ent_dir   = fullfile(base_dir, '06_Entregables');
if ~exist(ent_dir, 'dir'), mkdir(ent_dir); end

raw  = load(fullfile(datos_dir, 'datos_taller3.mat'));
part = load(fullfile(datos_dir, 'particion_datos.mat'));
red  = load(fullfile(res_dir, 'red_base_10_60_1.mat'));
p4   = load(fullfile(res_dir, 'poda_obs_punto4.mat'));

X_tr = part.X_tr; y_tr = double(part.y_tr);
X_val = part.X_val; y_val = double(part.y_val);
X_te = part.X_te_int; y_te = double(part.y_te_int);
X_test_norm = part.X_test_norm;
N_test = size(X_test_norm, 1);
[P, d] = size(X_tr); M = 60; n = 721;
X_rep = repelem(X_tr', M, 1);
mask_podables = true(n, 1); mask_podables(601:660) = false; mask_podables(721) = false;
max_podas = 655;

% Proporcion teorica de Clase 1 en x_test estimada por mezcla de medias en x_9
mu0_x9 = mean(raw.x_train(raw.y_train == 0, 9));
mu1_x9 = mean(raw.x_train(raw.y_train == 1, 9));
prop_c1_teorica_x9 = 100 * (mean(raw.x_test(:, 9)) - mu0_x9) / (mu1_x9 - mu0_x9);

fprintf('=========================================================================================\n');
fprintf('  PASO 5: EVALUACION EXHAUSTIVA DE LAS 2 REDES CANDIDATAS EN X_TEST (P = %d)\n', N_test);
fprintf('=========================================================================================\n\n');

%% 2. Definicion de las 2 Redes Candidatas (Candidata 1: trainbr vs Candidata 2: trainlm)
candidatas = struct('folder', {'Candidata_1_trainbr', 'Candidata_2_trainlm'}, ...
                    'tag',    {'Red #1: trainbr (tanh+lin)', 'Red #2: trainlm (sig+sig)'}, ...
                    'idx_top3', {1, 2}, ...
                    'alpha',   {1e-3, 1e-4});

res_candidatas = struct();

for c_idx = 1:2
    cand_dir = fullfile(ent_dir, candidatas(c_idx).folder);
    if ~exist(cand_dir, 'dir'), mkdir(cand_dir); end
    
    idx_t3   = candidatas(c_idx).idx_top3;
    red_obs  = p4.resultados_top3_obs(idx_t3);
    mod_base = red_obs.modelo_base;
    w_base   = mod_base.w_vec;
    ah       = mod_base.act_hid;
    ao       = mod_base.act_out;
    alpha_r  = candidatas(c_idx).alpha;
    
    fprintf('*****************************************************************************************\n');
    fprintf('  PROCESANDO %s -> Carpeta: 06_Entregables/%s/\n', candidatas(c_idx).tag, candidatas(c_idx).folder);
    fprintf('*****************************************************************************************\n');
    
    %% 2.1 Trayectorias de los 3 metodos de poda sobre esta candidata (para su Fig 9 propia)
    if c_idx == 1
        tr_mag  = p4.traj_mag;
        tr_obd  = p4.traj_obd;
        tr_puro = p4.traj_obs_puro;
    else
        tr_mag  = ejecutar_poda_cand(w_base, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, ah, ao, mask_podables, max_podas, 'mag', alpha_r);
        tr_obd  = ejecutar_poda_cand(w_base, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, ah, ao, mask_podables, max_podas, 'obd', alpha_r);
        tr_puro = ejecutar_poda_cand(w_base, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, ah, ao, mask_podables, max_podas, 'obs_puro', alpha_r);
    end
    
    %% 2.2 Parte A: Experimento de Precision IEEE-754 y las 4 Modalidades de Poda
    X_m_init   = calc_sens_cand(w_base, X_tr, y_tr, X_rep, d, M, ah, ao, true(n,1));
    H_inv_init = woodbury_hessiana_inv(X_m_init, alpha_r);
    
    residuos_ieee = zeros(n, 1);
    for q = 1:n
        if ~mask_podables(q), continue; end
        w_tmp = w_base - (w_base(q) / H_inv_init(q, q)) * H_inv_init(:, q);
        residuos_ieee(q) = abs(w_tmp(q));
    end
    n_no_cero_ieee = sum(residuos_ieee > 0);
    max_res_ieee   = max(residuos_ieee);
    
    K_demo = 50;
    % Modalidad 1A: Sin exclusion por mascara y sin proyeccion de H^-1
    w_1a = w_base; q_1a = zeros(K_demo,1); L_1a = zeros(K_demo,1); wb_1a = zeros(K_demo,1); u_1a = zeros(K_demo,1);
    % Modalidad 1B: Con exclusion por valor exacto (w==0), pero sin proyeccion de H^-1
    w_1b = w_base; q_1b = zeros(K_demo,1); L_1b = zeros(K_demo,1); wb_1b = zeros(K_demo,1); u_1b = zeros(K_demo,1);
    % Modalidad 1C: Con exclusion por mascara y con proyeccion de H^-1, pero SIN forzado explicito w(q*)=0 ni w.*mask
    w_1c = w_base; q_1c = zeros(K_demo,1); L_1c = zeros(K_demo,1); wb_1c = zeros(K_demo,1); u_1c = zeros(K_demo,1);
    mask_1c = true(n, 1); H_inv_1c = H_inv_init; res_1c_qstar = zeros(K_demo,1);
    % Modalidad 2: Con Triple Candado completo (exclusion por mascara, proyeccion H^-1, w(q*)=0 y w=w.*mask)
    w_2  = w_base; q_2  = zeros(K_demo,1); L_2  = zeros(K_demo,1); wb_2  = zeros(K_demo,1); u_2  = zeros(K_demo,1); nnz_2 = zeros(K_demo,1);
    mask_2 = true(n, 1); H_inv_2 = H_inv_init;
    
    for k = 1:K_demo
        % Modalidad 1A
        L_vec_a = (w_1a.^2) ./ (2 * max(diag(H_inv_init), 1e-15)); L_vec_a(~mask_podables) = inf;
        [L_1a(k), q_sa] = min(L_vec_a); q_1a(k) = q_sa; wb_1a(k) = w_1a(q_sa);
        w_1a = w_1a - (w_1a(q_sa) / H_inv_init(q_sa, q_sa)) * H_inv_init(:, q_sa);
        u_1a(k) = length(unique(q_1a(1:k)));
        
        % Modalidad 1B
        L_vec_b = (w_1b.^2) ./ (2 * max(diag(H_inv_init), 1e-15)); L_vec_b(~mask_podables) = inf; L_vec_b(w_1b == 0) = inf;
        [L_1b(k), q_sb] = min(L_vec_b); q_1b(k) = q_sb; wb_1b(k) = w_1b(q_sb);
        w_1b = w_1b - (w_1b(q_sb) / H_inv_init(q_sb, q_sb)) * H_inv_init(:, q_sb);
        u_1b(k) = length(unique(q_1b(1:k)));
        
        % Modalidad 1C
        L_vec_c = (w_1c.^2) ./ (2 * max(diag(H_inv_1c), 1e-15));
        L_vec_c(~(mask_1c & mask_podables)) = inf;
        [L_1c(k), q_sc] = min(L_vec_c); q_1c(k) = q_sc; wb_1c(k) = w_1c(q_sc);
        col_qc = H_inv_1c(:, q_sc); h_qqc = H_inv_1c(q_sc, q_sc);
        w_1c = w_1c - (w_1c(q_sc) / h_qqc) * col_qc;
        res_1c_qstar(k) = abs(w_1c(q_sc));
        H_inv_1c = H_inv_1c - (col_qc * col_qc') / h_qqc;
        mask_1c(q_sc) = false;
        u_1c(k) = length(unique(q_1c(1:k)));
        
        % Modalidad 2
        L_vec_2 = (w_2.^2) ./ (2 * max(diag(H_inv_2), 1e-15));
        L_vec_2(~(mask_2 & mask_podables)) = inf;
        [L_2(k), q_s2] = min(L_vec_2); q_2(k) = q_s2; wb_2(k) = w_2(q_s2);
        col_q = H_inv_2(:, q_s2); h_qq = H_inv_2(q_s2, q_s2);
        w_2 = w_2 - (w_2(q_s2) / h_qq) * col_q;
        H_inv_2 = H_inv_2 - (col_q * col_q') / h_qq;
        mask_2(q_s2) = false; w_2(q_s2) = 0; w_2 = w_2 .* double(mask_2);
        u_2(k) = length(unique(q_2(1:k)));
        nnz_2(k) = 660 - sum(abs(w_2(mask_podables)) == 0);
    end
    
    trazas_ieee = struct('q_1a', q_1a, 'L_1a', L_1a, 'wb_1a', wb_1a, 'u_1a', u_1a, ...
                         'q_1b', q_1b, 'L_1b', L_1b, 'wb_1b', wb_1b, 'u_1b', u_1b, ...
                         'q_1c', q_1c, 'L_1c', L_1c, 'wb_1c', wb_1c, 'u_1c', u_1c, 'res_1c_qstar', res_1c_qstar, ...
                         'q_2',  q_2,  'L_2',  L_2,  'wb_2',  wb_2,  'u_2',  u_2,  'nnz_2', nnz_2);
    
    fprintf('  [IEEE-754] Residuos ~= 0 en %d/660 pesos (%.1f%%%%), max |w_nuevo(q*)| = %.3e\n', ...
        n_no_cero_ieee, 100*n_no_cero_ieee/660, max_res_ieee);
    fprintf('  [IEEE-754] Modalidades: 1A elimina %d peso (q*=%d); 1B oscila en %d pesos (q*=%d, %d); 1C residuo ~1e-16; Mod 2 Candado: %d unicos\n', ...
        u_1a(end), q_1a(1), u_1b(end), q_1b(1), q_1b(2), u_2(end));
    
    %% 2.3 Parte B: Estimacion de Error Ajustado por Prevalencia y Simulacion en x_test
    w_opt    = red_obs.w_opt;
    mask_opt = red_obs.mask_opt;
    arq_c    = red_obs.info_arq;
    W1 = reshape(w_opt(1:600), M, d); b1 = w_opt(601:660);
    W2 = reshape(w_opt(661:720), 1, M); b2 = w_opt(721);
    w_vec = w_opt; mask_activos = mask_opt;
    mu_x = part.mu_x; sigma_x = part.sigma_x;
    act_hid = ah; act_out = ao;
    
    idx_neuronas_activas = arq_c.idx_ocultas_activas;
    W1_compacta = W1(idx_neuronas_activas, :);
    b1_compacta = b1(idx_neuronas_activas);
    W2_compacta = W2(idx_neuronas_activas);
    b2_compacta = b2;
    
    % Evaluacion en Val y Test_Int para intervalos de confianza Wilson y Bootstrap
    [~, err_val_c, auc_val_c, o_val_c] = eval_red_cand(w_opt, X_val, y_val, d, M, ah, ao);
    [E_te_c, err_te_c, auc_te_c, o_te_c] = eval_red_cand(w_opt, X_te, y_te, d, M, ah, ao);
    err_vec_te = double((o_te_c >= 0.5) ~= y_te);
    p_hat_te = mean(err_vec_te); N_te = length(y_te); z_95 = 1.95996;
    ci_wilson_te = 100 * ((p_hat_te + z_95^2/(2*N_te) + [-1, 1]*z_95*sqrt((p_hat_te*(1-p_hat_te) + z_95^2/(4*N_te))/N_te)) / (1 + z_95^2/N_te));
    err_vec_oos = [double((o_val_c >= 0.5) ~= y_val); err_vec_te];
    p_hat_oos = mean(err_vec_oos); N_oos = length(err_vec_oos);
    ci_wilson_oos = 100 * ((p_hat_oos + z_95^2/(2*N_oos) + [-1, 1]*z_95*sqrt((p_hat_oos*(1-p_hat_oos) + z_95^2/(4*N_oos))/N_oos)) / (1 + z_95^2/N_oos));
    rng(2026, 'twister'); B_boot = 5000;
    boot_te  = 100 * mean(err_vec_te(randi(N_te, N_te, B_boot)), 1);
    boot_oos = 100 * mean(err_vec_oos(randi(N_oos, N_oos, B_boot)), 1);
    ci_boot_te  = prctile(boot_te, [2.5, 97.5]);
    ci_boot_oos = prctile(boot_oos, [2.5, 97.5]);
    
    % Propagacion hacia adelante en x_test (forma completa 60x10 y forma compacta M_act x 10)
    o_test_cont = forward_mlp(W1, b1, W2, b2, X_test_norm, ah, ao);
    y_test_pred = double(o_test_cont >= 0.5);
    
    o_test_comp = forward_mlp(W1_compacta, b1_compacta, W2_compacta, b2_compacta, X_test_norm, ah, ao);
    y_test_comp = double(o_test_comp >= 0.5);
    diff_comp_max = max(abs(o_test_cont - o_test_comp));
    assert(diff_comp_max < 1e-14 && all(y_test_pred == y_test_comp), 'Error: Discrepancia entre red completa y compacta');
    
    % Comparacion contra su propia red base de 660 pesos sinapticos
    W1_b = reshape(w_base(1:600), M, d); b1_b = w_base(601:660);
    W2_b = reshape(w_base(661:720), 1, M); b2_b = w_base(721);
    o_test_base = forward_mlp(W1_b, b1_b, W2_b, b2_b, X_test_norm, ah, ao);
    y_test_base = double(o_test_base >= 0.5);
    concordancia_base_pct = 100 * mean(y_test_pred == y_test_base);
    corr_out_base         = corr(o_test_cont, o_test_base);
    
    n_pred_c0 = sum(y_test_pred == 0);
    n_pred_c1 = sum(y_test_pred == 1);
    prop_c1_pct = 100 * n_pred_c1 / N_test;
    
    % Estimador ajustado de error por cambio de proporcion de clases (Inversor Rogan-Gladen)
    y_oos      = [y_val; y_te];
    o_oos      = [o_val_c; o_te_c];
    y_oos_pred = double(o_oos >= 0.5);
    TP_oos = sum(y_oos_pred == 1 & y_oos == 1);
    FN_oos = sum(y_oos_pred == 0 & y_oos == 1);
    FP_oos = sum(y_oos_pred == 1 & y_oos == 0);
    TN_oos = sum(y_oos_pred == 0 & y_oos == 0);
    TPR_oos = TP_oos / (TP_oos + FN_oos);
    FNR_oos = FN_oos / (TP_oos + FN_oos);
    FPR_oos = FP_oos / (FP_oos + TN_oos);
    TNR_oos = TN_oos / (FP_oos + TN_oos);
    
    q_test = mean(y_test_pred == 1);
    p1_est = (q_test - FPR_oos) / (TPR_oos - FPR_oos);
    p1_est = max(0, min(1, p1_est));
    err_ajustado_pct = 100 * (p1_est * (1 - TPR_oos) + (1 - p1_est) * FPR_oos);
    
    rng(2026, 'twister'); B_boot_aj = 2000;
    boot_err_aj = zeros(B_boot_aj, 1); boot_p1_aj = zeros(B_boot_aj, 1);
    for b = 1:B_boot_aj
        idx_b = randi(N_oos, N_oos, 1);
        yb = y_oos(idx_b); pb = y_oos_pred(idx_b);
        tp_b = sum(pb == 1 & yb == 1); fn_b = sum(pb == 0 & yb == 1);
        fp_b = sum(pb == 1 & yb == 0); tn_b = sum(pb == 0 & yb == 0);
        tpr_b = tp_b / max(tp_b + fn_b, 1); fpr_b = fp_b / max(fp_b + tn_b, 1);
        den_b = tpr_b - fpr_b; if abs(den_b) < 1e-4, den_b = sign(den_b)*1e-4 + (den_b==0)*1e-4; end
        p1_b = max(0, min(1, (q_test - fpr_b) / den_b));
        boot_p1_aj(b) = p1_b;
        boot_err_aj(b) = 100 * (p1_b * (1 - tpr_b) + (1 - p1_b) * fpr_b);
    end
    ci_boot_err_aj = prctile(boot_err_aj, [2.5, 97.5]);
    ci_boot_p1_aj  = prctile(boot_p1_aj, [2.5, 97.5]);
    
    fprintf('  [Simulacion x_test] C0: %d (%.2f%%%%) | C1: %d (%.2f%%%%) [Teorica x_9: %.2f%%%%, |Desv| = %.2f%%%%]\n', ...
        n_pred_c0, 100*n_pred_c0/N_test, n_pred_c1, prop_c1_pct, prop_c1_teorica_x9, abs(prop_c1_pct - prop_c1_teorica_x9));
    fprintf('  [Prevalencia Rogan-Gladen] p1_est = %.2f%%%% (IC 95%%%% Boot: [%.2f%%%%, %.2f%%%%]) | Err_Ajustado = %.2f%%%% (IC 95%%%% Boot: [%.2f%%%%, %.2f%%%%])\n', ...
        100*p1_est, 100*ci_boot_p1_aj(1), 100*ci_boot_p1_aj(2), err_ajustado_pct, ci_boot_err_aj(1), ci_boot_err_aj(2));
    fprintf('  [Equivalencia Compacta] 10-60-1 vs 10-%d-1: Error Max = %.2e (100%%%% identicas)\n', length(idx_neuronas_activas), diff_comp_max);
    fprintf('  [Fidelidad Red Base] Concordancia con su red de 660 pesos: %.2f%%%% (%d/%d, r = %.6f)\n', ...
        concordancia_base_pct, sum(y_test_pred == y_test_base), N_test, corr_out_base);
    
    %% 2.4 Parte C: Guardado y Auditoria desde Cero de los Entregables Oficiales (OBS_1340.MAT)
    OBS_1340 = y_test_pred;
    OBS_13_40 = y_test_pred;
    etiquetas_prueba = y_test_pred;
    OBS_30 = y_test_pred;
    
    save(fullfile(cand_dir, 'OBS_1340.MAT'), 'OBS_1340');
    save(fullfile(cand_dir, 'OBS_13_40.mat'), 'OBS_13_40', 'etiquetas_prueba');
    save(fullfile(cand_dir, 'OBS_30.MAT'), 'OBS_30');
    
    if strcmp(ah, 'tanh'), desc_ah = 'tangente hiperbolica: tanh(z)'; else, desc_ah = 'sigmoide logistica: 1 ./ (1 + exp(-z))'; end
    if strcmp(ao, 'lin'),  desc_ao = 'lineal identidad: z';           else, desc_ao = 'sigmoide logistica: 1 ./ (1 + exp(-z))'; end
    
    arquitectura_final = struct( ...
        'candidata', candidatas(c_idx).folder, ...
        'algoritmo_entrenamiento', mod_base.algo, ...
        'configuracion_base', mod_base.config_name, ...
        'seed', mod_base.seed, ...
        'descripcion', sprintf('MLP 10-60-1 (%s) podado con OBS Puro (0%%%% reentrenamiento) a %d neuronas activas y %d pesos sinapticos', mod_base.algo, length(idx_neuronas_activas), arq_c.n_sinapticos), ...
        'estructura_original', [10, 60, 1], ...
        'estructura_efectiva_podada', [10, length(idx_neuronas_activas), 1], ...
        'n_parametros_originales', 721, ...
        'n_pesos_sinapticos_originales', 660, ...
        'n_pesos_sinapticos_eliminados_obs', 660 - arq_c.n_sinapticos, ...
        'n_pesos_sinapticos_activos', arq_c.n_sinapticos, ...
        'n_biases_activos', length(idx_neuronas_activas) + 1, ...
        'n_parametros_totales_activos', sum(mask_activos), ...
        'porcentaje_compresion_sinaptica', 100 * (660 - arq_c.n_sinapticos) / 660, ...
        'idx_neuronas_ocultas_activas', idx_neuronas_activas, ...
        'conexiones_activas_por_entrada', arq_c.conexiones_por_var, ...
        'norma_l1_por_entrada', arq_c.l1_por_var, ...
        'activacion_oculta', desc_ah, ...
        'activacion_salida', desc_ao, ...
        'umbral_decision', 0.5, ...
        'error_val_pct', err_val_c, ...
        'error_estimado_holdout_pct', err_te_c, ...
        'auc_holdout', auc_te_c, ...
        'ic95_wilson_holdout_pct', ci_wilson_te, ...
        'ic95_bootstrap_holdout_pct', ci_boot_te, ...
        'error_combinado_oos_pct', 100 * p_hat_oos, ...
        'ic95_wilson_oos_pct', ci_wilson_oos, ...
        'TPR_oos', TPR_oos, ...
        'FNR_oos', FNR_oos, ...
        'FPR_oos', FPR_oos, ...
        'TNR_oos', TNR_oos, ...
        'q_pred_clase1_xtest', q_test, ...
        'prevalencia_estimada_clase1_xtest', p1_est, ...
        'ic95_boot_prevalencia_clase1', ci_boot_p1_aj, ...
        'error_ajustado_xtest_pct', err_ajustado_pct, ...
        'ic95_boot_error_ajustado_pct', ci_boot_err_aj);
    
    save(fullfile(cand_dir, 'pesos_1340.mat'), ...
        'W1', 'b1', 'W2', 'b2', 'w_vec', 'mask_activos', ...
        'W1_compacta', 'b1_compacta', 'W2_compacta', 'b2_compacta', 'idx_neuronas_activas', ...
        'mu_x', 'sigma_x', 'act_hid', 'act_out', 'arquitectura_final');
    
    save(fullfile(cand_dir, 'pesos_13_40.mat'), ...
        'W1', 'b1', 'W2', 'b2', 'w_vec', 'mask_activos', ...
        'W1_compacta', 'b1_compacta', 'W2_compacta', 'b2_compacta', 'idx_neuronas_activas', ...
        'mu_x', 'sigma_x', 'act_hid', 'act_out', 'arquitectura_final');
    
    save(fullfile(cand_dir, 'pesos_30.mat'), ...
        'W1', 'b1', 'W2', 'b2', 'w_vec', 'mask_activos', ...
        'W1_compacta', 'b1_compacta', 'W2_compacta', 'b2_compacta', 'idx_neuronas_activas', ...
        'mu_x', 'sigma_x', 'act_hid', 'act_out', 'arquitectura_final');
    
    % Verificacion desde cero leyendo OBS_1340.MAT y pesos_1340.mat recien guardados
    chk_obs   = load(fullfile(cand_dir, 'OBS_1340.MAT'));
    chk_pesos = load(fullfile(cand_dir, 'pesos_1340.mat'));
    f_names = fieldnames(chk_obs);
    assert(isscalar(f_names), 'Error: OBS_1340.MAT debe tener una unica variable');
    y_ent = chk_obs.(f_names{1});
    assert(isequal(size(y_ent), [13888, 1]) && isa(y_ent, 'double'), 'Error de dimensiones o tipo en OBS_1340.MAT');
    assert(all(y_ent == 0 | y_ent == 1) && ~any(isnan(y_ent)), 'Error: Valores fuera de {0, 1} en OBS_1340.MAT');
    
    X_te_rec = (raw.x_test - chk_pesos.mu_x) ./ chk_pesos.sigma_x;
    o_rec_60 = forward_mlp(chk_pesos.W1, chk_pesos.b1, chk_pesos.W2, chk_pesos.b2, X_te_rec, chk_pesos.act_hid, chk_pesos.act_out);
    o_rec_cp = forward_mlp(chk_pesos.W1_compacta, chk_pesos.b1_compacta, chk_pesos.W2_compacta, chk_pesos.b2_compacta, X_te_rec, chk_pesos.act_hid, chk_pesos.act_out);
    assert(isequal(y_ent, double(o_rec_60 >= 0.5)) && isequal(y_ent, double(o_rec_cp >= 0.5)), ...
        'Error fatal: pesos_1340.mat no reproduce exactamente OBS_1340.MAT');
    fprintf('  [Auditoria Entregables] %s/OBS_1340.MAT y pesos_1340.mat verificados desde cero [OK]\n', candidatas(c_idx).folder);
    
    %% 2.5 Generacion de las 4 Figuras Propias de la Candidata en su Carpeta
    generar_figuras_candidata(cand_dir, mod_base, red_obs, arq_c, W1, b1, W2, ...
        tr_mag, tr_obd, tr_puro, p4.resultados_top3_obs, ...
        err_te_c, ci_wilson_te, p_hat_oos, boot_te, boot_oos, ...
        trazas_ieee, K_demo, ...
        o_test_cont, y_test_pred, n_pred_c0, n_pred_c1, N_test, raw.x_test, concordancia_base_pct);
    fprintf('  [Figuras] fig9, fig10, fig11 y fig12 exportadas en 06_Entregables/%s/ [OK]\n\n', candidatas(c_idx).folder);
    
    % Si es Candidata 1 (Red Campeona), replicar entregables en 06_Entregables/ y en la raiz del proyecto
    if c_idx == 1
        archivos_rep = {'OBS_1340.MAT', 'OBS_13_40.mat', 'OBS_30.MAT', ...
                        'pesos_1340.mat', 'pesos_13_40.mat', 'pesos_30.mat'};
        for f_k = 1:length(archivos_rep)
            copyfile(fullfile(cand_dir, archivos_rep{f_k}), fullfile(ent_dir, archivos_rep{f_k}));
            copyfile(fullfile(cand_dir, archivos_rep{f_k}), fullfile(base_dir, archivos_rep{f_k}));
        end
        copyfile(fullfile(cand_dir, 'fig11_efecto_precision_matlab_candado.png'), fullfile(fig_dir, 'fig11_efecto_precision_matlab_candado.png'));
        copyfile(fullfile(cand_dir, 'fig12_simulacion_final_xtest_OBS30.png'), fullfile(fig_dir, 'fig12_simulacion_final_xtest_OBS30.png'));
    end
    
    % Guardar resumen de la candidata en struct
    res_candidatas(c_idx).folder            = candidatas(c_idx).folder;
    res_candidatas(c_idx).algo              = mod_base.algo;
    res_candidatas(c_idx).config_name       = mod_base.config_name;
    res_candidatas(c_idx).seed              = mod_base.seed;
    res_candidatas(c_idx).n_ocultas         = length(idx_neuronas_activas);
    res_candidatas(c_idx).idx_ocultas       = idx_neuronas_activas;
    res_candidatas(c_idx).n_sinapticos      = arq_c.n_sinapticos;
    res_candidatas(c_idx).podados           = 660 - arq_c.n_sinapticos;
    res_candidatas(c_idx).compresion_pct    = 100 * (660 - arq_c.n_sinapticos) / 660;
    res_candidatas(c_idx).E_tr              = red_obs.E_tr;
    res_candidatas(c_idx).err_tr            = red_obs.err_tr;
    res_candidatas(c_idx).E_val             = red_obs.E_val;
    res_candidatas(c_idx).err_val           = err_val_c;
    res_candidatas(c_idx).E_te              = E_te_c;
    res_candidatas(c_idx).err_te_antes      = mod_base.err_te_int;
    res_candidatas(c_idx).err_te_despues    = err_te_c;
    res_candidatas(c_idx).auc_te            = auc_te_c;
    res_candidatas(c_idx).ci_wilson_te      = ci_wilson_te;
    res_candidatas(c_idx).ci_boot_te        = ci_boot_te;
    res_candidatas(c_idx).err_oos           = 100 * p_hat_oos;
    res_candidatas(c_idx).ci_wilson_oos     = ci_wilson_oos;
    res_candidatas(c_idx).TPR_oos           = TPR_oos;
    res_candidatas(c_idx).FNR_oos           = FNR_oos;
    res_candidatas(c_idx).FPR_oos           = FPR_oos;
    res_candidatas(c_idx).TNR_oos           = TNR_oos;
    res_candidatas(c_idx).q_test            = q_test;
    res_candidatas(c_idx).p1_est            = p1_est;
    res_candidatas(c_idx).ci_boot_p1        = ci_boot_p1_aj;
    res_candidatas(c_idx).err_ajustado      = err_ajustado_pct;
    res_candidatas(c_idx).ci_boot_err_aj    = ci_boot_err_aj;
    res_candidatas(c_idx).n_pred_c0         = n_pred_c0;
    res_candidatas(c_idx).n_pred_c1         = n_pred_c1;
    res_candidatas(c_idx).prop_c1_pct       = prop_c1_pct;
    res_candidatas(c_idx).desv_c1_teorica   = abs(prop_c1_pct - prop_c1_teorica_x9);
    res_candidatas(c_idx).concordancia_base = concordancia_base_pct;
    res_candidatas(c_idx).corr_out_base     = corr_out_base;
    res_candidatas(c_idx).n_no_cero_ieee    = n_no_cero_ieee;
    res_candidatas(c_idx).max_res_ieee      = max_res_ieee;
    res_candidatas(c_idx).trazas_ieee       = trazas_ieee;
    res_candidatas(c_idx).y_test_pred       = y_test_pred;
    res_candidatas(c_idx).o_test_cont       = o_test_cont;
    res_candidatas(c_idx).conexiones_var    = arq_c.conexiones_por_var;
    res_candidatas(c_idx).l1_var            = arq_c.l1_por_var;
end

%% 3. Comparativa Cruzada Integral sobre x_test (P = 13,888) incluyendo Red #3 (trainscg)
red3_obs = p4.resultados_top3_obs(3);
W1_r3 = reshape(red3_obs.w_opt(1:600), M, d); b1_r3 = red3_obs.w_opt(601:660);
W2_r3 = reshape(red3_obs.w_opt(661:720), 1, M); b2_r3 = red3_obs.w_opt(721);
o_test_r3 = forward_mlp(W1_r3, b1_r3, W2_r3, b2_r3, X_test_norm, red3_obs.modelo_base.act_hid, red3_obs.modelo_base.act_out);
y_test_r3 = double(o_test_r3 >= 0.5);

y_c1 = res_candidatas(1).y_test_pred;
y_c2 = res_candidatas(2).y_test_pred;
concordancia_c1_c2_pct = 100 * mean(y_c1 == y_c2);
concordancia_c1_r3_pct = 100 * mean(y_c1 == y_test_r3);
concordancia_c2_r3_pct = 100 * mean(y_c2 == y_test_r3);
corr_c1_c2_out         = corr(res_candidatas(1).o_test_cont, res_candidatas(2).o_test_cont);

% Voto mayoritario de las 3 redes finalistas podadas con OBS Puro (trainbr, trainlm, trainscg)
y_consenso_3 = double((y_c1 + y_c2 + y_test_r3) >= 2);
acuerdo_c1_consenso = 100 * mean(y_c1 == y_consenso_3);
acuerdo_c2_consenso = 100 * mean(y_c2 == y_consenso_3);
acuerdo_c3_consenso = 100 * mean(y_test_r3 == y_consenso_3);

fprintf('=========================================================================================\n');
fprintf('  TABLA COMPARATIVA INTEGRAL: CANDIDATA 1 (trainbr) vs CANDIDATA 2 (trainlm)\n');
fprintf('=========================================================================================\n');
fprintf('Metrica / Criterio                          | Candidata 1 (trainbr)     | Candidata 2 (trainlm)\n');
fprintf('-----------------------------------------------------------------------------------------\n');
fprintf('Configuracion y Activaciones                | %-25s | %-25s\n', res_candidatas(1).config_name, res_candidatas(2).config_name);
fprintf('Pesos Sinapticos Podados (Compresion %%)     | %3d / 660 (%6.2f%%%%)       | %3d / 660 (%6.2f%%%%)\n', ...
    res_candidatas(1).podados, res_candidatas(1).compresion_pct, res_candidatas(2).podados, res_candidatas(2).compresion_pct);
fprintf('Arquitectura Efectiva Sobreviviente         | 10 - %2d - 1 (%2d sin.)     | 10 - %2d - 1 (%2d sin.)\n', ...
    res_candidatas(1).n_ocultas, res_candidatas(1).n_sinapticos, res_candidatas(2).n_ocultas, res_candidatas(2).n_sinapticos);
fprintf('MSE Entrenamiento (E_tr) / Error Train      | %.4f / %5.2f%%%%           | %.4f / %5.2f%%%%\n', ...
    res_candidatas(1).E_tr, res_candidatas(1).err_tr, res_candidatas(2).E_tr, res_candidatas(2).err_tr);
fprintf('MSE Validacion (E_val) / Error Val          | %.4f / %5.2f%%%%           | %.4f / %5.2f%%%%\n', ...
    res_candidatas(1).E_val, res_candidatas(1).err_val, res_candidatas(2).E_val, res_candidatas(2).err_val);
fprintf('Error Test Interno Antes -> Despues OBS     | %5.2f%%%% -> %5.2f%%%% (%+5.2f%%%%) | %5.2f%%%% -> %5.2f%%%% (%+5.2f%%%%)\n', ...
    res_candidatas(1).err_te_antes, res_candidatas(1).err_te_despues, res_candidatas(1).err_te_despues - res_candidatas(1).err_te_antes, ...
    res_candidatas(2).err_te_antes, res_candidatas(2).err_te_despues, res_candidatas(2).err_te_despues - res_candidatas(2).err_te_antes);
fprintf('AUC ROC en Prueba Interna Hold-out (P=240)  | %.4f                    | %.4f\n', res_candidatas(1).auc_te, res_candidatas(2).auc_te);
fprintf('IC 95%%%% Wilson Hold-out (P=240)              | [%5.2f%%%%, %5.2f%%%%]          | [%5.2f%%%%, %5.2f%%%%]\n', ...
    res_candidatas(1).ci_wilson_te(1), res_candidatas(1).ci_wilson_te(2), res_candidatas(2).ci_wilson_te(1), res_candidatas(2).ci_wilson_te(2));
fprintf('Error Combinado Fuera de Muestra (P=480)    | %5.2f%%%% [%5.2f%%%%, %5.2f%%%%]   | %5.2f%%%% [%5.2f%%%%, %5.2f%%%%]\n', ...
    res_candidatas(1).err_oos, res_candidatas(1).ci_wilson_oos(1), res_candidatas(1).ci_wilson_oos(2), ...
    res_candidatas(2).err_oos, res_candidatas(2).ci_wilson_oos(1), res_candidatas(2).ci_wilson_oos(2));
fprintf('Prevalencia Estimada Clase 1 en x_test      | %5.2f%%%% [%5.2f%%%%, %5.2f%%%%]   | %5.2f%%%% [%5.2f%%%%, %5.2f%%%%]\n', ...
    100*res_candidatas(1).p1_est, 100*res_candidatas(1).ci_boot_p1(1), 100*res_candidatas(1).ci_boot_p1(2), ...
    100*res_candidatas(2).p1_est, 100*res_candidatas(2).ci_boot_p1(1), 100*res_candidatas(2).ci_boot_p1(2));
fprintf('Error Estimado Ajustado en x_test (Rogan-G) | %5.2f%%%% [%5.2f%%%%, %5.2f%%%%]   | %5.2f%%%% [%5.2f%%%%, %5.2f%%%%]\n', ...
    res_candidatas(1).err_ajustado, res_candidatas(1).ci_boot_err_aj(1), res_candidatas(1).ci_boot_err_aj(2), ...
    res_candidatas(2).err_ajustado, res_candidatas(2).ci_boot_err_aj(1), res_candidatas(2).ci_boot_err_aj(2));
fprintf('Predicciones Clase 1 en x_test (Teor:52.71%%)| %4d (%5.2f%%%%, Desv:%.2f%%%%) | %4d (%5.2f%%%%, Desv:%.2f%%%%)\n', ...
    res_candidatas(1).n_pred_c1, res_candidatas(1).prop_c1_pct, res_candidatas(1).desv_c1_teorica, ...
    res_candidatas(2).n_pred_c1, res_candidatas(2).prop_c1_pct, res_candidatas(2).desv_c1_teorica);
fprintf('Fidelidad vs su Red Base 660p en x_test     | %5.2f%%%% (r = %.4f)       | %5.2f%%%% (r = %.4f)\n', ...
    res_candidatas(1).concordancia_base, res_candidatas(1).corr_out_base, ...
    res_candidatas(2).concordancia_base, res_candidatas(2).corr_out_base);
fprintf('Acuerdo con Consenso de las 3 Finalistas    | %5.2f%%%%                    | %5.2f%%%%\n', acuerdo_c1_consenso, acuerdo_c2_consenso);
fprintf('Residuos IEEE-754 ~= 0 en los 660 pesos     | %3d / 660 (%.1f%%%%)         | %3d / 660 (%.1f%%%%)\n', ...
    res_candidatas(1).n_no_cero_ieee, 100*res_candidatas(1).n_no_cero_ieee/660, ...
    res_candidatas(2).n_no_cero_ieee, 100*res_candidatas(2).n_no_cero_ieee/660);
fprintf('-----------------------------------------------------------------------------------------\n');
fprintf('Concordancia directa C1 (trainbr) vs C2 (trainlm) en x_test : %.2f%%%% (%d / %d coinciden, r = %.4f)\n', ...
    concordancia_c1_c2_pct, sum(y_c1 == y_c2), N_test, corr_c1_c2_out);
fprintf('Concordancia C1 (trainbr) vs Red #3 (trainscg) en x_test    : %.2f%%%% (%d / %d coinciden)\n', ...
    concordancia_c1_r3_pct, sum(y_c1 == y_test_r3), N_test);
fprintf('Concordancia C2 (trainlm) vs Red #3 (trainscg) en x_test    : %.2f%%%% (%d / %d coinciden)\n', ...
    concordancia_c2_r3_pct, sum(y_c2 == y_test_r3), N_test);
fprintf('=========================================================================================\n');

% Variables IEEE directas de la Candidata 1 para guardado plano en comparativa_simulacion_paso5.mat
q_1a  = res_candidatas(1).trazas_ieee.q_1a;
L_1a  = res_candidatas(1).trazas_ieee.L_1a;
wb_1a = res_candidatas(1).trazas_ieee.wb_1a;
u_1a  = res_candidatas(1).trazas_ieee.u_1a;
q_1b  = res_candidatas(1).trazas_ieee.q_1b;
L_1b  = res_candidatas(1).trazas_ieee.L_1b;
wb_1b = res_candidatas(1).trazas_ieee.wb_1b;
u_1b  = res_candidatas(1).trazas_ieee.u_1b;
q_1c  = res_candidatas(1).trazas_ieee.q_1c;
L_1c  = res_candidatas(1).trazas_ieee.L_1c;
wb_1c = res_candidatas(1).trazas_ieee.wb_1c;
u_1c  = res_candidatas(1).trazas_ieee.u_1c;
q_2   = res_candidatas(1).trazas_ieee.q_2;
L_2   = res_candidatas(1).trazas_ieee.L_2;
wb_2  = res_candidatas(1).trazas_ieee.wb_2;
u_2   = res_candidatas(1).trazas_ieee.u_2;
nnz_2 = res_candidatas(1).trazas_ieee.nnz_2;

save(fullfile(res_dir, 'comparativa_simulacion_paso5.mat'), ...
    'res_candidatas', 'prop_c1_teorica_x9', 'concordancia_c1_c2_pct', ...
    'concordancia_c1_r3_pct', 'concordancia_c2_r3_pct', 'corr_c1_c2_out', ...
    'acuerdo_c1_consenso', 'acuerdo_c2_consenso', 'acuerdo_c3_consenso', 'y_consenso_3', ...
    'q_1a', 'L_1a', 'wb_1a', 'u_1a', ...
    'q_1b', 'L_1b', 'wb_1b', 'u_1b', ...
    'q_1c', 'L_1c', 'wb_1c', 'u_1c', ...
    'q_2',  'L_2',  'wb_2',  'u_2',  'nnz_2');

%% =========================================================================
%% FUNCIONES AUXILIARES (DISCIPLINA PONYTAIL)
%% =========================================================================

function o_col = forward_mlp(W1, b1, W2, b2, X, act_h, act_o)
    Z1 = W1 * X' + b1;
    if strcmp(act_h, 'tanh') || startsWith(act_h, 'tan'), A1 = tanh(Z1);
    else, A1 = 1 ./ (1 + exp(-Z1)); end
    z2 = W2 * A1 + b2;
    if strcmp(act_o, 'sig') || startsWith(act_o, 'sig'), o = 1 ./ (1 + exp(-z2));
    else, o = z2; end
    o_col = o';
end

function [E_mse, err_pct, auc_val, o_col] = eval_red_cand(w, X, y, d, M, act_h, act_o)
    W1 = reshape(w(1:600), M, d); b1 = w(601:660);
    W2 = reshape(w(661:720), 1, M); b2 = w(721);
    o_col = forward_mlp(W1, b1, W2, b2, X, act_h, act_o);
    E_mse = 0.5 * mean((y - o_col).^2);
    err_pct = 100 * mean((o_col >= 0.5) ~= y);
    if nargout > 2
        [~, idx] = sort(o_col, 'descend');
        ys = y(idx);
        tpr = [0; cumsum(ys == 1) / sum(ys == 1)];
        fpr = [0; cumsum(ys == 0) / sum(ys == 0)];
        auc_val = trapz(fpr, tpr);
    end
end

function [X_mat, e_tr] = calc_sens_cand(w, X, y, X_rep, d, M, act_h, act_o, mask_act)
    w = w .* double(mask_act);
    P = size(X, 1);
    W1 = reshape(w(1:600), M, d); b1 = w(601:660);
    W2 = reshape(w(661:720), 1, M); b2 = w(721);
    Z1 = W1 * X' + b1;
    if strcmp(act_h, 'tanh'), A1 = tanh(Z1); dA1 = 1 - A1.^2;
    else, A1 = 1 ./ (1 + exp(-Z1)); dA1 = A1 .* (1 - A1); end
    z2 = W2 * A1 + b2;
    if strcmp(act_o, 'sig'), o = 1 ./ (1 + exp(-z2)); do = o .* (1 - o);
    else, o = z2; do = ones(1, P); end
    delta1 = (W2' * do) .* dA1;
    X_mat = [repmat(delta1, d, 1) .* X_rep; delta1; A1 .* do; do] .* double(mask_act);
    e_tr  = y' - o;
end

function traj = ejecutar_poda_cand(w_init, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, act_h, act_o, mask_podables, max_podas, modo, alpha)
    P = size(X_tr, 1); n = length(w_init);
    w = w_init; mask_act = true(n, 1);
    E_tr = zeros(max_podas+1, 1); err_tr = zeros(max_podas+1, 1);
    E_val= zeros(max_podas+1, 1); err_val= zeros(max_podas+1, 1);
    E_te = zeros(max_podas+1, 1); err_te = zeros(max_podas+1, 1);
    [E_tr(1), err_tr(1)]   = eval_red_cand(w, X_tr, y_tr, d, M, act_h, act_o);
    [E_val(1), err_val(1)] = eval_red_cand(w, X_val, y_val, d, M, act_h, act_o);
    [E_te(1), err_te(1)]   = eval_red_cand(w, X_te, y_te, d, M, act_h, act_o);
    X_m = calc_sens_cand(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
    if strcmp(modo, 'obd')
        H_m = (X_m * X_m') / P;
    else
        H_inv = woodbury_hessiana_inv(X_m, alpha);
    end
    for step = 1:max_podas
        rem_syn = sum(mask_act & mask_podables);
        if strcmp(modo, 'obd') && rem_syn <= 180 && mod(step, 10) == 0
            X_m = calc_sens_cand(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
            H_m = (X_m * X_m') / P;
        elseif strcmp(modo, 'obs_puro') && rem_syn <= 180 && mod(step, 5) == 0
            act_idx = find(mask_act);
            X_m = calc_sens_cand(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
            H_inv = zeros(n, n);
            H_inv(act_idx, act_idx) = woodbury_hessiana_inv(X_m(act_idx, :), alpha);
        end
        if strcmp(modo, 'mag')
            L = abs(w);
        elseif strcmp(modo, 'obd')
            L = 0.5 * (diag(H_m) + alpha) .* (w.^2);
        else
            diag_Hinv = max(diag(H_inv), 1e-15);
            L = (w.^2) ./ (2 * diag_Hinv);
        end
        L(~(mask_act & mask_podables)) = inf;
        [~, q_star] = min(L);
        if strcmp(modo, 'obs_puro')
            col_q = H_inv(:, q_star); h_qq = H_inv(q_star, q_star);
            w = w - (w(q_star) / h_qq) * col_q;
            H_inv = H_inv - (col_q * col_q') / h_qq;
        end
        mask_act(q_star) = false; w(q_star) = 0; w = w .* double(mask_act);
        [E_tr(step+1), err_tr(step+1)] = eval_red_cand(w, X_tr, y_tr, d, M, act_h, act_o);
        [E_val(step+1), err_val(step+1)] = eval_red_cand(w, X_val, y_val, d, M, act_h, act_o);
        [E_te(step+1), err_te(step+1)]   = eval_red_cand(w, X_te, y_te, d, M, act_h, act_o);
    end
    traj.E_tr = E_tr; traj.err_tr = err_tr;
    traj.E_val = E_val; traj.err_val = err_val;
    traj.E_te = E_te; traj.err_te = err_te;
end

function generar_figuras_candidata(cand_dir, mod_base, red_obs, arq_c, W1, b1, W2, ...
    tr_mag, tr_obd, tr_puro, top3_obs, ...
    err_te_f, ci_wilson_te, p_hat_oos, boot_te, boot_oos, ...
    trazas_ieee, K_demo, ...
    o_test_cont, y_test_pred, n_pred_c0, n_pred_c1, N_test, x_test_raw, concordancia_base_pct)

    max_podas = length(tr_mag.E_tr) - 1;
    x_pod = 0:max_podas;
    idx_opt_plot = red_obs.idx_opt - 1;
    tr_sel = red_obs.trayectoria;
    d = 10;
    
    wb_1a = trazas_ieee.wb_1a; wb_1b = trazas_ieee.wb_1b; wb_1c = trazas_ieee.wb_1c; wb_2 = trazas_ieee.wb_2;
    u_1a  = trazas_ieee.u_1a;  u_1b  = trazas_ieee.u_1b;  u_1c  = trazas_ieee.u_1c;  u_2  = trazas_ieee.u_2;
    q_1a  = trazas_ieee.q_1a;  q_1b  = trazas_ieee.q_1b;  q_1c  = trazas_ieee.q_1c;

    %% FIGURA 9: Curvas de Poda OBS Puro vs OBD vs Magnitud de esta Candidata
    f9 = figure('Visible', 'off', 'Position', [50, 50, 1360, 740], 'Color', 'w');
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile; hold on;
    plot(x_pod, tr_mag.E_tr, '-.', 'LineWidth', 1.8, 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Poda por Magnitud');
    plot(x_pod, tr_obd.E_tr, '--', 'LineWidth', 1.8, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Optimal Brain Damage (OBD)');
    plot(x_pod, tr_puro.E_tr, '-', 'LineWidth', 2.2, 'Color', [0.0000 0.4470 0.7410], 'DisplayName', 'OBS Puro (0%% Reentrenamiento)');
    xline(idx_opt_plot, 'k--', 'LineWidth', 1.4, 'DisplayName', sprintf('Parada OBS (%d sin. activos)', arq_c.n_sinapticos));
    grid on; xlim([0 max_podas]); ylim([max(0.035, min(tr_puro.E_tr)*0.9), 0.16]);
    xlabel('Número de Pesos Sinápticos Eliminados (de 660)', 'FontSize', 10);
    ylabel('Error Cuadrático Medio E_{train}', 'FontSize', 10);
    title(sprintf('Trayectoria en Entrenamiento (%s, 0 a 655 Pesos Podados)', mod_base.config_name), 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'northwest', 'FontSize', 8.5); hold off;

    nexttile; hold on;
    rg_zoom = (480:645) + 1;
    plot(x_pod(rg_zoom), tr_mag.E_val(rg_zoom), '-.', 'LineWidth', 1.8, 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Magnitud (Val)');
    plot(x_pod(rg_zoom), tr_obd.E_val(rg_zoom), '--', 'LineWidth', 1.8, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'OBD (Val)');
    plot(x_pod(rg_zoom), tr_puro.E_val(rg_zoom), '-', 'LineWidth', 2.2, 'Color', [0.0000 0.4470 0.7410], 'DisplayName', 'OBS Puro (Val)');
    plot(idx_opt_plot, tr_sel.E_val(idx_opt_plot+1), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y', ...
        'DisplayName', sprintf('Punto Óptimo (%d sinápticos)', arq_c.n_sinapticos));
    grid on; xlim([480 645]); ylim([0.055 0.125]);
    xlabel('Pesos Sinápticos Eliminados (Región Estructural: 180 a 15 Restantes)', 'FontSize', 10);
    ylabel('MSE en Validación (E_{val})', 'FontSize', 10);
    title('Zoom Región Crítica: Estabilidad Superior de OBS frente a OBD y Magnitud', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'northwest', 'FontSize', 8.5); hold off;

    nexttile; hold on;
    plot(x_pod(rg_zoom), tr_mag.err_val(rg_zoom), '-.', 'LineWidth', 1.6, 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Magnitud');
    plot(x_pod(rg_zoom), tr_obd.err_val(rg_zoom), '--', 'LineWidth', 1.6, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'OBD');
    plot(x_pod(rg_zoom), tr_puro.err_val(rg_zoom), '-', 'LineWidth', 2.2, 'Color', [0.0000 0.4470 0.7410], 'DisplayName', 'OBS Puro');
    plot(idx_opt_plot, tr_sel.err_val(idx_opt_plot+1), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y', ...
        'DisplayName', sprintf('Óptimo Val (%.2f%%%%)', red_obs.err_val));
    grid on; xlim([480 645]); ylim([15 30]);
    xlabel('Número de Pesos Sinápticos Eliminados', 'FontSize', 10);
    ylabel('Error de Clasificación en Validación (%%)', 'FontSize', 10);
    title('Efecto de la Poda en el Error de Generalización (Validación)', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'northwest', 'FontSize', 8.5); hold off;

    nexttile; hold on;
    plot(x_pod(rg_zoom), tr_sel.err_tr(rg_zoom), '-', 'LineWidth', 1.8, 'Color', [0.0000 0.4470 0.7410], 'DisplayName', 'Entrenamiento (P=1120)');
    plot(x_pod(rg_zoom), tr_sel.err_val(rg_zoom), '--', 'LineWidth', 1.8, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Validación (P=240)');
    plot(x_pod(rg_zoom), tr_sel.err_te(rg_zoom), '-.', 'LineWidth', 2.2, 'Color', [0.2000 0.6500 0.3000], 'DisplayName', 'Prueba Interna Hold-out (P=240)');
    xline(idx_opt_plot, 'k--', 'LineWidth', 1.4, ...
        'DisplayName', sprintf('Parada OBS: %d neuronas (%d sin.)', arq_c.n_ocultas_activas, arq_c.n_sinapticos));
    plot(idx_opt_plot, tr_sel.err_te(idx_opt_plot+1), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.65 0.3], 'HandleVisibility', 'off');
    grid on; xlim([480 645]); ylim([11 25]);
    xlabel('Número de Pesos Sinápticos Eliminados', 'FontSize', 10);
    ylabel('Error de Clasificación (%%)', 'FontSize', 10);
    title(sprintf('Generalización OBS (%s): %d Neuronas y %d Pesos (Err_{TeInt} = %.2f%%%%)', ...
        mod_base.algo, arq_c.n_ocultas_activas, arq_c.n_sinapticos, err_te_f), 'FontSize', 10.5, 'FontWeight', 'bold');
    legend('Location', 'northwest', 'FontSize', 8.5); hold off;
    exportgraphics(f9, fullfile(cand_dir, 'fig9_curvas_poda_obs_vs_obd_mag.png'), 'Resolution', 200);
    close(f9);

    %% FIGURA 10: Arquitectura Final Podada y Estimacion en x_test de esta Candidata
    f10 = figure('Visible', 'off', 'Position', [50, 50, 1360, 740], 'Color', 'w');
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    idx_n_act = arq_c.idx_ocultas_activas;
    mat_arq_vis = [W1(idx_n_act, :), b1(idx_n_act), W2(idx_n_act)'];
    imagesc(mat_arq_vis);
    colormap(gca, parula); cb_a = colorbar; cb_a.Label.String = 'Valor del Peso Sináptico / Bias';
    hold on;
    [r_zero, c_zero] = find(abs(mat_arq_vis) == 0);
    plot(c_zero, r_zero, 'wo', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
    xline(10.5, 'w--', 'LineWidth', 1.5); xline(11.5, 'w--', 'LineWidth', 1.5);
    xticks(1:12); xticklabels({'x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','x_9','x_{10}','b_1','W_2^T'});
    yticks(1:length(idx_n_act));
    yticklabels(arrayfun(@(id) sprintf('N_{%d}', id), idx_n_act, 'UniformOutput', false));
    xlabel('Entradas (x_1..x_{10}), Bias Oculto (b_1) y Peso de Salida (W_2)', 'FontSize', 10);
    ylabel(sprintf('Neuronas Ocultas Sobrevivientes (%d de 60)', arq_c.n_ocultas_activas), 'FontSize', 10);
    title(sprintf('Mapa de Conectividad Red Podada (%s: Puntos Negros = Pesos en 0)', mod_base.algo), 'FontSize', 11, 'FontWeight', 'bold');
    hold off;

    nexttile;
    [~, var_min_l1] = min(arq_c.l1_por_var);
    yyaxis left;
    bar(1:d, arq_c.conexiones_por_var, 0.55, 'FaceColor', [0.0000 0.4470 0.7410], 'FaceAlpha', 0.75);
    ylabel(sprintf('Número de Conexiones Activas (de %d)', arq_c.n_ocultas_activas), 'FontSize', 10);
    ylim([0 arq_c.n_ocultas_activas + 2]);
    yyaxis right;
    plot(1:d, arq_c.l1_por_var, '-s', 'LineWidth', 2.0, 'MarkerSize', 7, ...
        'Color', [0.8500 0.3250 0.0980], 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
    ylabel('Importancia Acumulada \Sigma_j |W_{1,ji}|', 'FontSize', 10);
    ylim([0 max(arq_c.l1_por_var)*1.22]);
    grid on; xticks(1:d);
    xticklabels(arrayfun(@(i) sprintf('x_{%d}', i), 1:d, 'UniformOutput', false));
    xlabel(sprintf('Variable de Entrada (Menor peso L_1 tras poda: x_{%d})', var_min_l1), 'FontSize', 10);
    title(sprintf('Selección Implícita de Características por OBS (%s)', mod_base.algo), 'FontSize', 11, 'FontWeight', 'bold');
    legend({'Conexiones Activas', 'Norma L_1 de Pesos'}, 'Location', 'northwest', 'FontSize', 9);

    nexttile;
    err_antes_3 = arrayfun(@(x) x.modelo_base.err_te_int, top3_obs);
    err_val_3   = [top3_obs.err_val];
    err_te_3    = [top3_obs.err_te];
    mat_3r      = [err_antes_3; err_val_3; err_te_3]';
    b_comp = bar(1:3, mat_3r, 'grouped');
    b_comp(1).FaceColor = [0.60 0.60 0.60];
    b_comp(2).FaceColor = [0.8500 0.3250 0.0980];
    b_comp(3).FaceColor = [0.2000 0.6500 0.3000];
    lbl_3r = arrayfun(@(r) sprintf('Red %d: %s (%dN, %dp)', r, top3_obs(r).modelo_base.algo, ...
        top3_obs(r).n_ocultas_act, top3_obs(r).rem_syn_opt), 1:3, 'UniformOutput', false);
    grid on; xticks(1:3); xticklabels(lbl_3r);
    ylabel('Error de Clasificación (%%)', 'FontSize', 10);
    ylim([0 max(mat_3r(:))*1.28]);
    title('Comparativa de las 3 Redes Candidatas: Base (660 sin.) vs Podada OBS Puro', 'FontSize', 11, 'FontWeight', 'bold');
    legend({'Test Int. Antes (660 sin.)', 'Validación Después OBS', 'Test Int. Después OBS'}, 'Location', 'northwest', 'FontSize', 8.5);
    for k_b = 1:3
        xtips = b_comp(k_b).XEndPoints; ytips = b_comp(k_b).YEndPoints;
        labels_b = arrayfun(@(v) sprintf('%.2f%%%%', v), ytips, 'UniformOutput', false);
        text(xtips, ytips + 0.45, labels_b, 'HorizontalAlignment', 'center', 'FontSize', 8.0, 'FontWeight', 'bold');
    end

    nexttile; hold on;
    histogram(boot_te, 22, 'Normalization', 'pdf', 'FaceColor', [0.2000 0.6500 0.3000], 'FaceAlpha', 0.55, 'EdgeColor', 'w');
    histogram(boot_oos, 22, 'Normalization', 'pdf', 'FaceColor', [0.0000 0.4470 0.7410], 'FaceAlpha', 0.55, 'EdgeColor', 'w');
    xline(err_te_f, 'k-', 'LineWidth', 2.0);
    xline(ci_wilson_te(1), 'k--', 'LineWidth', 1.2); xline(ci_wilson_te(2), 'k--', 'LineWidth', 1.2);
    grid on; xlabel('Error de Clasificación Estimado en x\_test (%%)', 'FontSize', 10);
    ylabel('Densidad Bootstrap (B = 5000 réplicas)', 'FontSize', 10);
    title(sprintf('Estimación Error en x\\_test (%s): %.2f%%%% (IC 95%%%%: [%.2f%%%%, %.2f%%%%])', ...
        mod_base.algo, err_te_f, ci_wilson_te(1), ci_wilson_te(2)), 'FontSize', 11, 'FontWeight', 'bold');
    legend({sprintf('Hold-out P=240 (Media=%.2f%%%%)', err_te_f), ...
            sprintf('Fuera de Muestra P=480 (Media=%.2f%%%%)', 100*p_hat_oos), ...
            'Estimador Puntual Hold-out', 'IC 95%%%% Wilson'}, 'Location', 'northeast', 'FontSize', 8.5);
    hold off;
    exportgraphics(f10, fullfile(cand_dir, 'fig10_arquitectura_final_y_estimacion_error.png'), 'Resolution', 200);
    close(f10);

    %% FIGURA 11: Demostracion del Efecto de Precision IEEE-754 en esta Candidata
    f11 = figure('Visible', 'off', 'Position', [60, 60, 1340, 520], 'Color', 'w');
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile; hold on;
    iters_d = 1:25;
    semilogy(iters_d, max(abs(wb_1a(iters_d)), 1e-160), ':', 'LineWidth', 1.8, ...
        'Color', [0.4660 0.6740 0.1880], 'DisplayName', sprintf('Mod 1A: Sin Exclusión (oscila q*=%d)', q_1a(1)));
    semilogy(iters_d, max(abs(wb_1b(iters_d)), 1e-160), '-s', 'LineWidth', 2.0, 'MarkerSize', 6, ...
        'Color', [0.8500 0.3250 0.0980], 'MarkerFaceColor', [0.8500 0.3250 0.0980], ...
        'DisplayName', sprintf('Mod 1B: Sin Máscara (Resurrección oscila q*=%d y %d)', q_1b(1), q_1b(2)));
    semilogy(iters_d, max(abs(wb_1c(iters_d)), 1e-160), '-^', 'LineWidth', 1.8, 'MarkerSize', 5, ...
        'Color', [0.4940 0.1840 0.5560], 'MarkerFaceColor', [0.4940 0.1840 0.5560], ...
        'DisplayName', 'Mod 1C: Máscara y H^{-1} proy. (sin forzado w(q*)=0, residuo ~10^{-16})');
    semilogy(iters_d, abs(wb_2(iters_d)), '-o', 'LineWidth', 2.0, 'MarkerSize', 6, ...
        'Color', [0.0000 0.4470 0.7410], 'MarkerFaceColor', [0.0000 0.4470 0.7410], ...
        'DisplayName', 'Mod 2: Con Triple Candado (w(q*)=0 y w = w .* mask)');
    yline(eps, 'k--', 'LineWidth', 1.3, 'DisplayName', 'Épsilon de Máquina MATLAB (\epsilon = 2.22 \times 10^{-16})');
    grid on; xlim([1 25]); ylim([1e-160, 1e0]);
    xlabel('Iteración del Bucle de Poda OBS', 'FontSize', 10.5);
    ylabel('Magnitud del Peso Seleccionado |w_{q*}| (escala log)', 'FontSize', 10.5);
    title(sprintf('Trampa de Punto Flotante (%s): Resurrección de Pesos Podados', mod_base.algo), 'FontSize', 11.5, 'FontWeight', 'bold');
    legend('Location', 'southwest', 'FontSize', 8.0); hold off;

    nexttile; hold on;
    plot(1:K_demo, u_2, '-', 'LineWidth', 2.5, 'Color', [0.0000 0.4470 0.7410], ...
        'DisplayName', 'Mod 2: Con Candado Triple (50 pesos únicos eliminados)');
    plot(1:K_demo, u_1c, '-.', 'LineWidth', 2.0, 'Color', [0.4940 0.1840 0.5560], ...
        'DisplayName', sprintf('Mod 1C: Con Máscara y H^{-1} proy. (inicia q*=%d)', q_1c(1)));
    plot(1:K_demo, u_1b, '--', 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980], ...
        'DisplayName', sprintf('Mod 1B: Sin Máscara (Estancado en %d pesos: q*=%d, %d)', u_1b(end), q_1b(1), q_1b(2)));
    plot(1:K_demo, u_1a, ':', 'LineWidth', 2.5, 'Color', [0.4660 0.6740 0.1880], ...
        'DisplayName', sprintf('Mod 1A: Sin Exclusión (Estancado en %d peso: q*=%d)', u_1a(end), q_1a(1)));
    grid on; xlim([1 K_demo]); ylim([0 55]);
    xlabel('Iteración del Bucle de Poda OBS', 'FontSize', 10.5);
    ylabel('Número Acumulado de Pesos Únicos Eliminados', 'FontSize', 10.5);
    title(sprintf('Progreso Real de Poda (%s): Necesidad Obligatoria del Candado en Cero', mod_base.algo), 'FontSize', 11.5, 'FontWeight', 'bold');
    legend('Location', 'northwest', 'FontSize', 8.5); hold off;
    exportgraphics(f11, fullfile(cand_dir, 'fig11_efecto_precision_matlab_candado.png'), 'Resolution', 200);
    close(f11);

    %% FIGURA 12: Diagnostico de la Simulacion Final sobre x_test (OBS_1340.MAT)
    f12 = figure('Visible', 'off', 'Position', [60, 60, 1340, 520], 'Color', 'w');
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile; hold on;
    if strcmp(mod_base.act_out, 'sig')
        edges_p = linspace(0, 1, 45);
        xl_lims = [0, 1];
        x_lbl_str = 'Salida Continua Sigmoide o_{test} \in (0, 1)';
    else
        q_lo = min(-0.35, prctile(o_test_cont, 0.5));
        q_hi = max(1.35,  prctile(o_test_cont, 99.5));
        edges_p = linspace(q_lo, q_hi, 48);
        xl_lims = [q_lo, q_hi];
        x_lbl_str = 'Salida Continua Lineal de la Red Podada o_{test}';
    end
    histogram(o_test_cont(y_test_pred == 0), edges_p, 'FaceColor', [0.8500 0.3250 0.0980], 'FaceAlpha', 0.65, 'EdgeColor', 'w');
    histogram(o_test_cont(y_test_pred == 1), edges_p, 'FaceColor', [0.0000 0.4470 0.7410], 'FaceAlpha', 0.65, 'EdgeColor', 'w');
    xline(0.5, 'k--', 'LineWidth', 1.8);
    grid on; xlim(xl_lims);
    xlabel(x_lbl_str, 'FontSize', 10.5);
    ylabel('Número de Muestras en x\_test (Total = 13,888)', 'FontSize', 10.5);
    title(sprintf('Distribución en x\\_test (%s: C0=%d [%.1f%%%%] | C1=%d [%.1f%%%%])', ...
        mod_base.algo, n_pred_c0, 100*n_pred_c0/N_test, n_pred_c1, 100*n_pred_c1/N_test), 'FontSize', 11.5, 'FontWeight', 'bold');
    legend({'Predicho Clase 0 (o_{test} < 0.5)', 'Predicho Clase 1 (o_{test} \geq 0.5)', 'Umbral 0.5'}, 'Location', 'north', 'FontSize', 9);
    hold off;

    nexttile; hold on;
    rng(30, 'twister');
    idx_sub_vis = randperm(N_test, 2500);
    x9_vis = x_test_raw(idx_sub_vis, 9);
    x6_vis = x_test_raw(idx_sub_vis, 6);
    y_vis  = y_test_pred(idx_sub_vis);
    scatter(x9_vis(y_vis == 0), x6_vis(y_vis == 0), 16, [0.8500 0.3250 0.0980], 'filled', 'MarkerFaceAlpha', 0.55);
    scatter(x9_vis(y_vis == 1), x6_vis(y_vis == 1), 16, [0.0000 0.4470 0.7410], 'filled', 'MarkerFaceAlpha', 0.55);
    grid on;
    xlabel('Variable x_9 en Unidades Originales (x\_test)', 'FontSize', 10.5);
    ylabel('Variable x_6 en Unidades Originales (x\_test)', 'FontSize', 10.5);
    title(sprintf('Separación No Lineal en x\\_test (%s: Acuerdo con Red 660p = %.2f%%%%)', mod_base.algo, concordancia_base_pct), ...
        'FontSize', 11.5, 'FontWeight', 'bold');
    legend({'Etiqueta OBS\_1340 = 0 (Clase 0)', 'Etiqueta OBS\_1340 = 1 (Clase 1)'}, 'Location', 'northeast', 'FontSize', 9);
    hold off;
    exportgraphics(f12, fullfile(cand_dir, 'fig12_simulacion_final_xtest_OBS30.png'), 'Resolution', 200);
    close(f12);
end
