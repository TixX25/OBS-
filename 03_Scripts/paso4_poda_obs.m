%% PASO 4: PODA OPTIMAL BRAIN SURGEON (OBS), COMPARATIVA DE 3 REDES Y RED OPTIMA
% Implementacion del ciclo destructivo de Optimal Brain Surgeon (Hassibi & Stork,
% NIPS 1993, Ecs. 5, 17), comparativa contra OBD y Magnitud, evaluacion sobre las
% 3 Redes Finalistas (trainbr, trainrp, trainlm) y estimacion del error en x_test.

clear; clc; close all;

%% 1. Carga de particiones y modelos base entrenados
base_dir  = fileparts(fileparts(mfilename('fullpath')));
if isempty(base_dir), base_dir = pwd; end
datos_dir = fullfile(base_dir, '02_Datos');
fig_dir   = fullfile(base_dir, '04_Figuras');
res_dir   = fullfile(base_dir, '05_Resultados_Parciales');

part = load(fullfile(datos_dir, 'particion_datos.mat'));
red  = load(fullfile(res_dir, 'red_base_10_60_1.mat'));

X_tr = part.X_tr; y_tr = part.y_tr;
X_val = part.X_val; y_val = part.y_val;
X_te = part.X_te_int; y_te = part.y_te_int;
[P, d] = size(X_tr); M = 60; n = 721;
X_rep = repelem(X_tr', M, 1);

% Mascara de los 660 pesos sinapticos podables (protegiendo los 61 biases b1 y b2)
mask_podables = true(n, 1);
mask_podables(601:660) = false;
mask_podables(721)     = false;
n_podables = sum(mask_podables); % 660 pesos sinapticos
max_podas  = 655;                % Podar desde 660 hasta dejar solo 5 pesos sinapticos

fprintf('=========================================================================================\n');
fprintf('  PASO 4: ALGORITMO DE PODA OPTIMAL BRAIN SURGEON (OBS) SOBRE LAS 3 REDES FINALISTAS\n');
fprintf('=========================================================================================\n');
fprintf('Red inicial: 10 - %d - 1 (n = %d parametros totales, %d pesos sinapticos podables)\n', M, n, n_podables);
fprintf('Recorrido de poda destructiva: de %d a %d pesos sinapticos (%d cortes secuenciales)\n\n', n_podables, n_podables-max_podas, max_podas);

%% 2. Experimento 1: Comparativa de Poda (Magnitud vs OBD vs OBS Puro vs OBS + Pulido) en Red Campeona (trainbr)
mod_camp  = red.mejor_modelo;
w_base    = mod_camp.w_vec;
ah_base   = mod_camp.act_hid;
ao_base   = mod_camp.act_out;
lam_base  = mod_camp.lambda;
alpha_ref = 1e-5;

fprintf('--- EXPERIMENTO 1: Comparacion de Metodos de Poda sobre Red Campeona (%s, Seed %d) ---\n', ...
    mod_camp.config_name, mod_camp.seed);
t_exp1 = tic;

traj_mag      = ejecutar_poda(w_base, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, ah_base, ao_base, mask_podables, max_podas, 'mag', alpha_ref, 0);
traj_obd      = ejecutar_poda(w_base, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, ah_base, ao_base, mask_podables, max_podas, 'obd', alpha_ref, 0);
traj_obs_puro = ejecutar_poda(w_base, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, ah_base, ao_base, mask_podables, max_podas, 'obs_puro', alpha_ref, 0);
traj_obs_pol  = ejecutar_poda(w_base, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, ah_base, ao_base, mask_podables, max_podas, 'obs_pol', alpha_ref, lam_base);

fprintf('Experimento 1 completado en %.2f segundos.\n', toc(t_exp1));
fprintf('Verificacion de triple candado numerico (w(q*)=0, mask y unicidad de indices podados): [OK]\n\n');

cortes_chk = [0, 500, 540, 570, 600, 620, 640];
fprintf('--------------------------------------------------------------------------------------------------------------------------------------------------\n');
fprintf('Podados (Rest) |   Magnitud: E_tr / Err_Val / Err_Te   |    OBD: E_tr / Err_Val / Err_Te     |  OBS Puro: E_tr / Err_Val / Err_Te  | OBS+Pol: E_tr / Err_Val / Err_Te\n');
fprintf('--------------------------------------------------------------------------------------------------------------------------------------------------\n');
for c_i = 1:length(cortes_chk)
    sc = cortes_chk(c_i);
    idx_s = sc + 1;
    fprintf(' %3d (%3d sin) |  %.4f / %5.2f%% / %5.2f%%          |  %.4f / %5.2f%% / %5.2f%%        |  %.4f / %5.2f%% / %5.2f%%        | %.4f / %5.2f%% / %5.2f%%\n', ...
        sc, n_podables - sc, ...
        traj_mag.E_tr(idx_s), traj_mag.err_val(idx_s), traj_mag.err_te(idx_s), ...
        traj_obd.E_tr(idx_s), traj_obd.err_val(idx_s), traj_obd.err_te(idx_s), ...
        traj_obs_puro.E_tr(idx_s), traj_obs_puro.err_val(idx_s), traj_obs_puro.err_te(idx_s), ...
        traj_obs_pol.E_tr(idx_s), traj_obs_pol.err_val(idx_s), traj_obs_pol.err_te(idx_s));
end
fprintf('--------------------------------------------------------------------------------------------------------------------------------------------------\n\n');

%% 3. Experimento 2A: Sensibilidad al Parametro Alpha en la Red Campeona (trainbr)
fprintf('--- EXPERIMENTO 2A: Sensibilidad de OBS al Parametro de Amortiguamiento Alpha (Red 1: trainbr) ---\n');
alphas_eval = [1e-6, 1e-5, 1e-4, 1e-3];
tabla_sens_alpha = struct();
for a_i = 1:length(alphas_eval)
    al = alphas_eval(a_i);
    tr_a = ejecutar_poda(w_base, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, ah_base, ao_base, mask_podables, max_podas, 'obs_puro', al, 0);
    rg = (545:630) + 1;
    [~, idx_b] = sortrows([round(tr_a.err_val(rg), 3), round(tr_a.E_val(rg), 3), -(rg')]);
    b_step = rg(idx_b(1));
    tabla_sens_alpha(a_i).alpha   = al;
    tabla_sens_alpha(a_i).podados = b_step - 1;
    tabla_sens_alpha(a_i).rem_syn = n_podables - (b_step - 1);
    tabla_sens_alpha(a_i).E_val   = tr_a.E_val(b_step);
    tabla_sens_alpha(a_i).err_val = tr_a.err_val(b_step);
    tabla_sens_alpha(a_i).err_te  = tr_a.err_te(b_step);
    fprintf('  Alpha = %.0e | Optimo Puro en %d podados (%d sinapticos activos): E_val=%.4f | Err_Val=%.2f%% | Err_TeInt=%.2f%%\n', ...
        al, b_step-1, n_podables-(b_step-1), tr_a.E_val(b_step), tr_a.err_val(b_step), tr_a.err_te(b_step));
end

%% 4. Experimento 2B: Comparativa de Poda OBS en las 3 Redes Finalistas (top3_modelos) y Top 5
fprintf('\n--- EXPERIMENTO 2B: Poda OBS sobre las 3 Redes Finalistas (trainbr, trainrp, trainlm) y Top 5 ---\n');
fprintf('--------------------------------------------------------------------------------------------------------------------------------------------------\n');
fprintf('Red | Algoritmo | Configuracion                 | Seed | Podados (Sin. Rem) | Neuronas Activas |  E_tr  | Err_Tr |  E_val | Err_Val |  E_te  | Err_TeInt\n');
fprintf('--------------------------------------------------------------------------------------------------------------------------------------------------\n');

resultados_top5_obs = struct();
for m_i = 1:5
    mk = red.top5_modelos(m_i);
    lam_k = max(mk.lambda, 5e-5);
    if m_i == 1
        tr_puro = traj_obs_puro;
        tr_pol  = traj_obs_pol;
    else
        tr_puro = ejecutar_poda(mk.w_vec, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, mk.act_hid, mk.act_out, mask_podables, max_podas, 'obs_puro', 1e-5, 0);
        tr_pol  = ejecutar_poda(mk.w_vec, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, mk.act_hid, mk.act_out, mask_podables, max_podas, 'obs_pol', 1e-5, lam_k);
    end
    
    % Seleccion en regimen de alta compresion (>= 550 podados) con fidelidad E_tr <= E_tr(1) + 0.0045
    rg_base = (550:625) + 1;
    rg_puro = rg_base(tr_puro.E_tr(rg_base) <= tr_puro.E_tr(1) + 0.0045); if isempty(rg_puro), rg_puro = rg_base; end
    rg_pol  = rg_base(tr_pol.E_tr(rg_base)  <= tr_pol.E_tr(1)  + 0.0045); if isempty(rg_pol),  rg_pol  = rg_base; end
    [~, ord_puro] = sortrows([round(tr_puro.err_val(rg_puro), 2), round(tr_puro.E_val(rg_puro), 3), -(rg_puro')]);
    [~, ord_pol]  = sortrows([round(tr_pol.err_val(rg_pol), 2),   round(tr_pol.E_val(rg_pol), 3),   -(rg_pol')]);
    idx_puro = rg_puro(ord_puro(1));
    idx_pol  = rg_pol(ord_pol(1));
    
    % Elegir entre obs_pol y obs_puro segun menor error en Validacion (desempatando por E_val)
    if (tr_pol.err_val(idx_pol) < tr_puro.err_val(idx_puro)) || ...
       (tr_pol.err_val(idx_pol) == tr_puro.err_val(idx_puro) && tr_pol.E_val(idx_pol) <= tr_puro.E_val(idx_puro))
        tr_k = tr_pol; idx_opt_k = idx_pol; variante_k = 'obs_pol';
    else
        tr_k = tr_puro; idx_opt_k = idx_puro; variante_k = 'obs_puro';
    end
    
    w_opt_k = tr_k.w_hist(:, idx_opt_k);
    [w_clean_k, mask_clean_k, info_arq_k] = limpiar_neuronas_muertas(w_opt_k, d, M, mk.act_hid);
    [E_tr_c, err_tr_c]   = eval_red_completa(w_clean_k, X_tr, y_tr, d, M, mk.act_hid, mk.act_out);
    [E_val_c, err_val_c] = eval_red_completa(w_clean_k, X_val, y_val, d, M, mk.act_hid, mk.act_out);
    [E_te_c, err_te_c, auc_te_c] = eval_red_completa(w_clean_k, X_te, y_te, d, M, mk.act_hid, mk.act_out);
    
    resultados_top5_obs(m_i).modelo_base   = mk;
    resultados_top5_obs(m_i).variante_obs  = variante_k;
    resultados_top5_obs(m_i).trayectoria   = tr_k;
    resultados_top5_obs(m_i).idx_opt       = idx_opt_k;
    resultados_top5_obs(m_i).podados_opt   = 660 - info_arq_k.n_sinapticos;
    resultados_top5_obs(m_i).rem_syn_opt   = info_arq_k.n_sinapticos;
    resultados_top5_obs(m_i).n_ocultas_act = info_arq_k.n_ocultas_activas;
    resultados_top5_obs(m_i).w_opt         = w_clean_k;
    resultados_top5_obs(m_i).mask_opt      = mask_clean_k;
    resultados_top5_obs(m_i).info_arq      = info_arq_k;
    resultados_top5_obs(m_i).E_tr          = E_tr_c;
    resultados_top5_obs(m_i).err_tr        = err_tr_c;
    resultados_top5_obs(m_i).E_val         = E_val_c;
    resultados_top5_obs(m_i).err_val       = err_val_c;
    resultados_top5_obs(m_i).E_te          = E_te_c;
    resultados_top5_obs(m_i).err_te        = err_te_c;
    resultados_top5_obs(m_i).auc_te        = auc_te_c;
    
    fprintf(' #%d | %-9s | %-29s | %4d |   %3d   (%3d rem)  |     %2d / 60      | %.4f | %5.2f%% | %.4f |  %5.2f%% | %.4f |   %5.2f%%\n', ...
        m_i, mk.algo, mk.config_name, mk.seed, 660 - info_arq_k.n_sinapticos, info_arq_k.n_sinapticos, ...
        info_arq_k.n_ocultas_activas, E_tr_c, err_tr_c, E_val_c, err_val_c, E_te_c, err_te_c);
end
fprintf('--------------------------------------------------------------------------------------------------------------------------------------------------\n\n');

resultados_top3_obs = resultados_top5_obs(1:3);

%% 5. Seleccion de la Red Optima Final Campeona y Analisis de su Estructura
red_optima     = resultados_top3_obs(1);
w_opt_final    = red_optima.w_opt;
mask_opt_final = red_optima.mask_opt;
arq_final      = red_optima.info_arq;

W1_opt = reshape(w_opt_final(1:600), M, d);
b1_opt = w_opt_final(601:660);
W2_opt = reshape(w_opt_final(661:720), 1, M);
b2_opt = w_opt_final(721);

fprintf('=========================================================================================\n');
fprintf('  ARQUITECTURA FINAL PODADA SELECCIONADA (CAMPEONA OBS: %s)\n', red_optima.modelo_base.algo);
fprintf('=========================================================================================\n');
fprintf('Configuracion base        : %s (Seed %d)\n', red_optima.modelo_base.config_name, red_optima.modelo_base.seed);
fprintf('Reduccion de neuronas     : De 60 a %d neuronas ocultas activas (%.1f%% de neuronas eliminadas)\n', ...
    arq_final.n_ocultas_activas, 100 * (60 - arq_final.n_ocultas_activas) / 60);
fprintf('Reduccion de pesos sinap. : De 660 a %d pesos sinapticos (%d eliminados = %.2f%% de compresion)\n', ...
    arq_final.n_sinapticos, 660 - arq_final.n_sinapticos, 100 * (660 - arq_final.n_sinapticos) / 660);
fprintf('Parametros totales no cero: De 721 a %d parametros (incluyendo %d biases activos = %.2f%% compresion total)\n\n', ...
    sum(mask_opt_final), sum(mask_opt_final) - arq_final.n_sinapticos, 100 * (721 - sum(mask_opt_final)) / 721);

fprintf('Conectividad por Variable de Entrada (x_1 .. x_10) en las %d Neuronas Ocultas Activas:\n', arq_final.n_ocultas_activas);
fprintf('-------------------------------------------------------------------------\n');
fprintf('Variable | Conexiones Activas (de %2d) | Norma L1 (sum|w|) | Norma L2\n', arq_final.n_ocultas_activas);
fprintf('-------------------------------------------------------------------------\n');
for j = 1:d
    fprintf('  x_%-2d   |          %2d (%5.1f%%)       |      %7.3f      |  %7.3f\n', ...
        j, arq_final.conexiones_por_var(j), 100*arq_final.conexiones_por_var(j)/arq_final.n_ocultas_activas, ...
        arq_final.l1_por_var(j), arq_final.l2_por_var(j));
end
fprintf('-------------------------------------------------------------------------\n\n');

%% 6. Estimacion Rigurosa del Error sobre el Conjunto Solucion (x_test)
[~, err_tr_f, auc_tr_f, o_tr_f]    = eval_red_completa(w_opt_final, X_tr, y_tr, d, M, ah_base, ao_base);
[~, err_val_f, auc_val_f, o_val_f] = eval_red_completa(w_opt_final, X_val, y_val, d, M, ah_base, ao_base);
[~, err_te_f, auc_te_f, o_te_f]    = eval_red_completa(w_opt_final, X_te, y_te, d, M, ah_base, ao_base);

err_vec_te = double((o_te_f >= 0.5) ~= y_te);
p_hat_te   = mean(err_vec_te);
se_te      = sqrt(p_hat_te * (1 - p_hat_te) / length(y_te));
z_95 = 1.95996;
N_te = length(y_te);
ci_wilson_te = 100 * ((p_hat_te + z_95^2/(2*N_te) + [-1, 1]*z_95*sqrt((p_hat_te*(1-p_hat_te) + z_95^2/(4*N_te))/N_te)) / (1 + z_95^2/N_te));

err_vec_oos = [double((o_val_f >= 0.5) ~= y_val); err_vec_te];
p_hat_oos   = mean(err_vec_oos);
N_oos       = length(err_vec_oos);
ci_wilson_oos = 100 * ((p_hat_oos + z_95^2/(2*N_oos) + [-1, 1]*z_95*sqrt((p_hat_oos*(1-p_hat_oos) + z_95^2/(4*N_oos))/N_oos)) / (1 + z_95^2/N_oos));

rng(2026, 'twister');
B_boot = 5000;
boot_te  = 100 * mean(err_vec_te(randi(N_te, N_te, B_boot)), 1);
boot_oos = 100 * mean(err_vec_oos(randi(N_oos, N_oos, B_boot)), 1);
ci_boot_te  = prctile(boot_te, [2.5, 97.5]);
ci_boot_oos = prctile(boot_oos, [2.5, 97.5]);

estimacion_solucion = struct('err_te_int', err_te_f, 'auc_te_int', auc_te_f, 'se_te_pct', 100*se_te, ...
    'ci_wilson_te', ci_wilson_te, 'ci_boot_te', ci_boot_te, 'boot_te', boot_te, ...
    'err_oos_comb', 100*p_hat_oos, 'ci_wilson_oos', ci_wilson_oos, 'ci_boot_oos', ci_boot_oos, 'boot_oos', boot_oos);

fprintf('=========================================================================================\n');
fprintf('  ESTIMACION DEL ERROR SOBRE EL CONJUNTO SOLUCION (x_test, P = 13888)\n');
fprintf('=========================================================================================\n');
fprintf('1. Estimador insesgado Hold-out (Prueba Interna, P = 240):\n');
fprintf('   -> Error de Clasificacion = %.2f%% (Exactitud = %.2f%%, AUC = %.4f, MSE = %.4f)\n', ...
    err_te_f, 100 - err_te_f, auc_te_f, red_optima.E_te);
fprintf('   -> IC 95%% Binomial (Wilson): [%.2f%%, %.2f%%] | IC 95%% Bootstrap: [%.2f%%, %.2f%%]\n', ...
    ci_wilson_te(1), ci_wilson_te(2), ci_boot_te(1), ci_boot_te(2));
fprintf('2. Estimador Combinado Fuera de Muestra (Val + Test_Int, P = 480):\n');
fprintf('   -> Error de Clasificacion = %.2f%% (Exactitud = %.2f%%)\n', 100*p_hat_oos, 100*(1 - p_hat_oos));
fprintf('   -> IC 95%% Binomial (Wilson): [%.2f%%, %.2f%%] | IC 95%% Bootstrap: [%.2f%%, %.2f%%]\n\n', ...
    ci_wilson_oos(1), ci_wilson_oos(2), ci_boot_oos(1), ci_boot_oos(2));

%% 7. Guardado de Resultados Parciales del Punto 4
save(fullfile(res_dir, 'poda_obs_punto4.mat'), ...
    'red_optima', 'w_opt_final', 'mask_opt_final', 'W1_opt', 'b1_opt', 'W2_opt', 'b2_opt', ...
    'arq_final', 'estimacion_solucion', 'traj_mag', 'traj_obd', 'traj_obs_puro', 'traj_obs_pol', ...
    'tabla_sens_alpha', 'resultados_top3_obs', 'resultados_top5_obs');

%% 8. FIGURA 9: Curvas de Poda Destructiva (OBS vs OBD vs Magnitud en Red Campeona: trainbr)
fig9 = figure('Name', 'Curvas de Poda OBS vs OBD vs Magnitud', 'Position', [50, 50, 1360, 740], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
x_pod = 0:max_podas;
idx_opt_plot = red_optima.idx_opt - 1;
tr_camp_plot = red_optima.trayectoria;

% Panel 1: Evolucion completa del MSE en Entrenamiento (0 a 655 pesos eliminados)
nexttile; hold on;
plot(x_pod, traj_mag.E_tr, '-.', 'LineWidth', 1.8, 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Poda por Magnitud');
plot(x_pod, traj_obd.E_tr, '--', 'LineWidth', 1.8, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Optimal Brain Damage (OBD)');
plot(x_pod, traj_obs_puro.E_tr, ':', 'LineWidth', 2.0, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'OBS Puro (sin reajuste)');
plot(x_pod, traj_obs_pol.E_tr, '-', 'LineWidth', 2.2, 'Color', [0.0000 0.4470 0.7410], 'DisplayName', 'OBS + Pulido Subespacio Activo');
xline(idx_opt_plot, 'k--', 'LineWidth', 1.4, 'DisplayName', sprintf('Parada OBS (%d sin. activos)', arq_final.n_sinapticos));
grid on; xlim([0 max_podas]); ylim([0.05 0.16]);
xlabel('Número de Pesos Sinápticos Eliminados (de 660)', 'FontSize', 10);
ylabel('Error Cuadrático Medio E_{train}', 'FontSize', 10);
title(sprintf('Trayectoria en Entrenamiento (%s, 0 a 655 Pesos Podados)', mod_camp.config_name), 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 8.5);
hold off;

% Panel 2: Zoom en la Region Critica de Poda Estructural (480 a 645 pesos eliminados) - MSE Validacion
nexttile; hold on;
rg_zoom = (480:645) + 1;
plot(x_pod(rg_zoom), traj_mag.E_val(rg_zoom), '-.', 'LineWidth', 1.8, 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Magnitud (Val)');
plot(x_pod(rg_zoom), traj_obd.E_val(rg_zoom), '--', 'LineWidth', 1.8, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'OBD (Val)');
plot(x_pod(rg_zoom), traj_obs_puro.E_val(rg_zoom), ':', 'LineWidth', 2.0, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'OBS Puro (Val)');
plot(x_pod(rg_zoom), traj_obs_pol.E_val(rg_zoom), '-', 'LineWidth', 2.2, 'Color', [0.0000 0.4470 0.7410], 'DisplayName', 'OBS + Pulido (Val)');
plot(idx_opt_plot, tr_camp_plot.E_val(idx_opt_plot+1), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y', ...
    'DisplayName', sprintf('Punto Óptimo (%d sinápticos)', arq_final.n_sinapticos));
grid on; xlim([480 645]); ylim([0.058 0.125]);
xlabel('Pesos Sinápticos Eliminados (Región Estructural: 180 a 15 Restantes)', 'FontSize', 10);
ylabel('MSE en Validación (E_{val})', 'FontSize', 10);
title('Zoom Región Crítica: Estabilidad Superior de OBS frente a OBD y Magnitud', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 8.5);
hold off;

% Panel 3: Error de Clasificacion (%) en Validacion en la Region Critica
nexttile; hold on;
plot(x_pod(rg_zoom), traj_mag.err_val(rg_zoom), '-.', 'LineWidth', 1.6, 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Magnitud');
plot(x_pod(rg_zoom), traj_obd.err_val(rg_zoom), '--', 'LineWidth', 1.6, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'OBD');
plot(x_pod(rg_zoom), traj_obs_puro.err_val(rg_zoom), ':', 'LineWidth', 1.8, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'OBS Puro');
plot(x_pod(rg_zoom), traj_obs_pol.err_val(rg_zoom), '-', 'LineWidth', 2.2, 'Color', [0.0000 0.4470 0.7410], 'DisplayName', 'OBS + Pulido');
plot(idx_opt_plot, tr_camp_plot.err_val(idx_opt_plot+1), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y', ...
    'DisplayName', sprintf('Óptimo Val (%.2f%%)', red_optima.err_val));
grid on; xlim([480 645]); ylim([15 30]);
xlabel('Número de Pesos Sinápticos Eliminados', 'FontSize', 10);
ylabel('Error de Clasificación en Validación (%)', 'FontSize', 10);
title('Efecto de la Poda en el Error de Generalización (Validación)', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 8.5);
hold off;

% Panel 4: Evolucion simultanea Train / Val / Test Interno para OBS y Seleccion de Tamano Apropiado
nexttile; hold on;
plot(x_pod(rg_zoom), tr_camp_plot.err_tr(rg_zoom), '-', 'LineWidth', 1.8, 'Color', [0.0000 0.4470 0.7410], 'DisplayName', 'Entrenamiento (P=1120)');
plot(x_pod(rg_zoom), tr_camp_plot.err_val(rg_zoom), '--', 'LineWidth', 1.8, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Validación (P=240)');
plot(x_pod(rg_zoom), tr_camp_plot.err_te(rg_zoom), '-.', 'LineWidth', 2.2, 'Color', [0.2000 0.6500 0.3000], 'DisplayName', 'Prueba Interna Hold-out (P=240)');
xline(idx_opt_plot, 'k--', 'LineWidth', 1.4, ...
    'DisplayName', sprintf('Parada OBS: %d neuronas (%d sin.)', arq_final.n_ocultas_activas, arq_final.n_sinapticos));
plot(idx_opt_plot, tr_camp_plot.err_te(idx_opt_plot+1), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.65 0.3], 'HandleVisibility', 'off');
grid on; xlim([480 645]); ylim([12 25]);
xlabel('Número de Pesos Sinápticos Eliminados', 'FontSize', 10);
ylabel('Error de Clasificación (%)', 'FontSize', 10);
title(sprintf('Generalización OBS (%s): %d Neuronas y %d Pesos (Err_{TeInt} = %.2f%%)', ...
    mod_camp.algo, arq_final.n_ocultas_activas, arq_final.n_sinapticos, red_optima.err_te), 'FontSize', 10.5, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 8.5);
hold off;

exportgraphics(fig9, fullfile(fig_dir, 'fig9_curvas_poda_obs_vs_obd_mag.png'), 'Resolution', 200);

%% 9. FIGURA 10: Arquitectura Final Podada, Comparativa de las 3 Redes Finalistas y Estimacion en x_test
fig10 = figure('Name', 'Arquitectura Final Podada y Estimacion Error', 'Position', [50, 50, 1360, 740], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Mapa de calor de los pesos de las neuronas ocultas activas (W1_act y W2_act)
nexttile;
idx_n_act = arq_final.idx_ocultas_activas;
mat_arq_vis = [W1_opt(idx_n_act, :), b1_opt(idx_n_act), W2_opt(idx_n_act)'];
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
ylabel(sprintf('Neuronas Ocultas Sobrevivientes (%d de 60)', arq_final.n_ocultas_activas), 'FontSize', 10);
title('Mapa de Conectividad de la Red Podada (Puntos Negros = Pesos en 0)', 'FontSize', 11, 'FontWeight', 'bold');
hold off;

% Panel 2: Importancia y Conectividad por Variable de Entrada en la Red Podada
nexttile;
[~, var_min_l1] = min(arq_final.l1_por_var);
yyaxis left;
bar(1:d, arq_final.conexiones_por_var, 0.55, 'FaceColor', [0.0000 0.4470 0.7410], 'FaceAlpha', 0.75);
ylabel(sprintf('Número de Conexiones Activas (de %d)', arq_final.n_ocultas_activas), 'FontSize', 10);
ylim([0 arq_final.n_ocultas_activas + 2]);
yyaxis right;
plot(1:d, arq_final.l1_por_var, '-s', 'LineWidth', 2.0, 'MarkerSize', 7, ...
    'Color', [0.8500 0.3250 0.0980], 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
ylabel('Importancia Acumulada \Sigma_j |W_{1,ji}|', 'FontSize', 10);
ylim([0 max(arq_final.l1_por_var)*1.22]);
grid on; xticks(1:d);
xticklabels(arrayfun(@(i) sprintf('x_{%d}', i), 1:d, 'UniformOutput', false));
xlabel(sprintf('Variable de Entrada (Menor peso L_1 tras poda: x_{%d})', var_min_l1), 'FontSize', 10);
title('Selección Implícita de Características por Optimal Brain Surgeon', 'FontSize', 11, 'FontWeight', 'bold');
legend({'Conexiones Activas', 'Norma L_1 de Pesos'}, 'Location', 'northwest', 'FontSize', 9);

% Panel 3: Comparativa Directa de las 3 Redes Finalistas (trainbr, trainrp, trainlm) Antes vs Despues de Poda OBS
nexttile;
err_antes_3 = arrayfun(@(x) x.modelo_base.err_te_int, resultados_top3_obs);
err_val_3   = [resultados_top3_obs.err_val];
err_te_3    = [resultados_top3_obs.err_te];
mat_3r      = [err_antes_3; err_val_3; err_te_3]';
b_comp = bar(1:3, mat_3r, 'grouped');
b_comp(1).FaceColor = [0.60 0.60 0.60];
b_comp(2).FaceColor = [0.8500 0.3250 0.0980];
b_comp(3).FaceColor = [0.2000 0.6500 0.3000];
lbl_3r = arrayfun(@(r) sprintf('Red %d: %s (%dN, %dp)', r, resultados_top3_obs(r).modelo_base.algo, ...
    resultados_top3_obs(r).n_ocultas_act, resultados_top3_obs(r).rem_syn_opt), 1:3, 'UniformOutput', false);
grid on; xticks(1:3); xticklabels(lbl_3r);
ylabel('Error de Clasificación (%)', 'FontSize', 10);
ylim([0 max(mat_3r(:))*1.28]);
title('Comparativa de las 3 Redes Finalistas: Base (660 sin.) vs Podada OBS', 'FontSize', 11, 'FontWeight', 'bold');
legend({'Test Int. Antes (660 sin.)', 'Validación Después OBS', 'Test Int. Después OBS'}, 'Location', 'northwest', 'FontSize', 8.5);
for k_b = 1:3
    xtips = b_comp(k_b).XEndPoints;
    ytips = b_comp(k_b).YEndPoints;
    labels_b = arrayfun(@(v) sprintf('%.2f%%', v), ytips, 'UniformOutput', false);
    text(xtips, ytips + 0.45, labels_b, 'HorizontalAlignment', 'center', 'FontSize', 8.0, 'FontWeight', 'bold');
end

% Panel 4: Distribucion Bootstrap del Error Estimado sobre el Conjunto Solucion (x_test)
nexttile; hold on;
histogram(boot_te, 22, 'Normalization', 'pdf', 'FaceColor', [0.2000 0.6500 0.3000], 'FaceAlpha', 0.55, 'EdgeColor', 'w');
histogram(boot_oos, 22, 'Normalization', 'pdf', 'FaceColor', [0.0000 0.4470 0.7410], 'FaceAlpha', 0.55, 'EdgeColor', 'w');
xline(err_te_f, 'k-', 'LineWidth', 2.0);
xline(ci_wilson_te(1), 'k--', 'LineWidth', 1.2); xline(ci_wilson_te(2), 'k--', 'LineWidth', 1.2);
grid on; xlabel('Error de Clasificación Estimado en x\_test (%)', 'FontSize', 10);
ylabel('Densidad Bootstrap (B = 5000 réplicas)', 'FontSize', 10);
title(sprintf('Estimación Error en Conjunto Solución: %.2f%% (IC 95%%: [%.2f%%, %.2f%%])', ...
    err_te_f, ci_wilson_te(1), ci_wilson_te(2)), 'FontSize', 11, 'FontWeight', 'bold');
legend({sprintf('Hold-out P=240 (Media=%.2f%%)', err_te_f), ...
        sprintf('Fuera de Muestra P=480 (Media=%.2f%%)', 100*p_hat_oos), ...
        'Estimador Puntual Hold-out', 'IC 95% Wilson'}, 'Location', 'northeast', 'FontSize', 8.5);
hold off;

exportgraphics(fig10, fullfile(fig_dir, 'fig10_arquitectura_final_y_estimacion_error.png'), 'Resolution', 200);
fprintf('Figuras fig9 y fig10 y archivo poda_obs_punto4.mat generados exitosamente.\n');

%% =========================================================================
%% FUNCIONES DEL MOTOR DE PODA OBS (DISCIPLINA PONYTAIL)
%% =========================================================================

function traj = ejecutar_poda(w_init, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, act_h, act_o, mask_podables, max_podas, modo, alpha, lam)
    % Ejecuta la poda secuencial registrando metricas y aplicando el triple candado en cero
    P = size(X_tr, 1); n = length(w_init);
    w = w_init;
    mask_act = true(n, 1);
    
    E_tr = zeros(max_podas+1, 1); err_tr = zeros(max_podas+1, 1);
    E_val= zeros(max_podas+1, 1); err_val= zeros(max_podas+1, 1);
    E_te = zeros(max_podas+1, 1); err_te = zeros(max_podas+1, 1);
    w_hist = zeros(n, max_podas+1);
    historial_q = zeros(max_podas, 1);
    
    [E_tr(1), err_tr(1)]   = eval_red_completa(w, X_tr, y_tr, d, M, act_h, act_o);
    [E_val(1), err_val(1)] = eval_red_completa(w, X_val, y_val, d, M, act_h, act_o);
    [E_te(1), err_te(1)]   = eval_red_completa(w, X_te, y_te, d, M, act_h, act_o);
    w_hist(:, 1) = w;
    
    X_m   = calc_sens_obs(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
    H_m   = (X_m * X_m') / P;
    H_inv = inv(H_m + alpha * eye(n));
    
    for step = 1:max_podas
        rem_syn = sum(mask_act & mask_podables);
        
        if strcmp(modo, 'obd') && rem_syn <= 180 && mod(step, 10) == 0
            X_m = calc_sens_obs(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
            H_m = (X_m * X_m') / P;
        elseif strcmp(modo, 'obs_puro') && rem_syn <= 180 && mod(step, 5) == 0
            act_idx = find(mask_act);
            X_m = calc_sens_obs(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
            H_sub = (X_m(act_idx, :) * X_m(act_idx, :)') / P;
            H_inv = zeros(n, n);
            H_inv(act_idx, act_idx) = inv(H_sub + alpha * eye(length(act_idx)));
        elseif strcmp(modo, 'obs_pol') && rem_syn <= 180 && mod(step, 5) == 0
            act_idx = find(mask_act);
            for lm_it = 1:2
                [X_m, e_tr] = calc_sens_obs(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
                J_act = X_m(act_idx, :);
                g_act = -(J_act * e_tr') / P + lam * w(act_idx);
                H_act = (J_act * J_act') / P + (lam + 2e-3) * eye(length(act_idx));
                w(act_idx) = w(act_idx) - H_act \ g_act;
                w = w .* double(mask_act);
            end
            X_m = calc_sens_obs(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
            H_sub = (X_m(act_idx, :) * X_m(act_idx, :)') / P;
            H_inv = zeros(n, n);
            H_inv(act_idx, act_idx) = inv(H_sub + alpha * eye(length(act_idx)));
        end
        
        if strcmp(modo, 'mag')
            L = abs(w);
        elseif strcmp(modo, 'obd')
            L = 0.5 * (diag(H_m) + alpha) .* (w.^2);
        else
            diag_Hinv = max(diag(H_inv), 1e-15);
            L = (w.^2) ./ (2 * diag_Hinv);
        end
        
        % Candado 1: Asignar L_q = +Inf a todos los pesos ya eliminados y a los biases
        L(~(mask_act & mask_podables)) = inf;
        [~, q_star] = min(L);
        historial_q(step) = q_star;
        
        % Ajuste quirurgico de pesos si el modo es OBS (Ec. 5 de Hassibi & Stork)
        if startsWith(modo, 'obs')
            col_q = H_inv(:, q_star);
            h_qq  = H_inv(q_star, q_star);
            w = w - (w(q_star) / h_qq) * col_q;
            H_inv = H_inv - (col_q * col_q') / h_qq;
        end
        
        % Candado 2 y 3 (Indicacion estricta del profesor): Fijar en cero y multiplicar por mascara
        mask_act(q_star) = false;
        w(q_star) = 0;
        w = w .* double(mask_act);
        assert(w(q_star) == 0 && ~mask_act(q_star), 'Error: Peso podado no quedo en cero estricto');
        
        [E_tr(step+1), err_tr(step+1)]   = eval_red_completa(w, X_tr, y_tr, d, M, act_h, act_o);
        if strcmp(modo, 'obs_pol') && rem_syn <= 180 && E_tr(step+1) > E_tr(step) + 0.004
            act_idx = find(mask_act);
            for lm_it = 1:2
                [X_m, e_tr] = calc_sens_obs(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
                J_act = X_m(act_idx, :);
                g_act = -(J_act * e_tr') / P + lam * w(act_idx);
                H_act = (J_act * J_act') / P + (lam + 2e-3) * eye(length(act_idx));
                w(act_idx) = w(act_idx) - H_act \ g_act;
                w = w .* double(mask_act);
            end
            X_m = calc_sens_obs(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
            H_sub = (X_m(act_idx, :) * X_m(act_idx, :)') / P;
            H_inv = zeros(n, n);
            H_inv(act_idx, act_idx) = inv(H_sub + alpha * eye(length(act_idx)));
            [E_tr(step+1), err_tr(step+1)] = eval_red_completa(w, X_tr, y_tr, d, M, act_h, act_o);
        end
        [E_val(step+1), err_val(step+1)] = eval_red_completa(w, X_val, y_val, d, M, act_h, act_o);
        [E_te(step+1), err_te(step+1)]   = eval_red_completa(w, X_te, y_te, d, M, act_h, act_o);
        w_hist(:, step+1) = w;
    end
    
    assert(length(unique(historial_q)) == max_podas, 'Error: Se intento podar un peso repetido');
    
    traj.E_tr = E_tr; traj.err_tr = err_tr;
    traj.E_val= E_val; traj.err_val= err_val;
    traj.E_te = E_te; traj.err_te = err_te;
    traj.w_hist = w_hist; traj.historial_q = historial_q;
end

function [X_mat, e_tr] = calc_sens_obs(w, X, y, X_rep, d, M, act_h, act_o, mask_act)
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

function [w_clean, mask_clean, info] = limpiar_neuronas_muertas(w, d, M, act_h)
    % Limpia pesos huerfanos de neuronas ocultas desconectadas (donde W2_j = 0 o toda fila W1_j = 0)
    W1 = reshape(w(1:600), M, d); b1 = w(601:660);
    W2 = reshape(w(661:720), 1, M); b2 = w(721);
    
    const_mask = (abs(W2') > 0) & ~any(abs(W1) > 0, 2);
    if any(const_mask)
        if strcmp(act_h, 'tanh'), a_const = tanh(b1(const_mask));
        else, a_const = 1 ./ (1 + exp(-b1(const_mask))); end
        b2 = b2 + W2(const_mask) * a_const;
    end
    
    act_n_mask = (abs(W2') > 0) & any(abs(W1) > 0, 2);
    W1_c = W1 .* act_n_mask;
    b1_c = b1 .* act_n_mask;
    W2_c = W2 .* (act_n_mask');
    w_clean = [W1_c(:); b1_c; W2_c(:); b2];
    mask_clean = (abs(w_clean) > 0);
    
    info.idx_ocultas_activas = find(act_n_mask);
    info.n_ocultas_activas   = sum(act_n_mask);
    info.n_W1_activos        = sum(abs(W1_c(:)) > 0);
    info.n_W2_activos        = sum(abs(W2_c(:)) > 0);
    info.n_sinapticos        = info.n_W1_activos + info.n_W2_activos;
    info.conexiones_por_var  = sum(abs(W1_c) > 0, 1);
    info.l1_por_var          = sum(abs(W1_c), 1);
    info.l2_por_var          = sqrt(sum(W1_c.^2, 1));
end

function [E_mse, err_pct, auc_val, o_col] = eval_red_completa(w, X, y, d, M, act_h, act_o)
    W1 = reshape(w(1:600), M, d); b1 = w(601:660);
    W2 = reshape(w(661:720), 1, M); b2 = w(721);
    Z1 = W1 * X' + b1;
    if strcmp(act_h, 'tanh'), A1 = tanh(Z1); else, A1 = 1 ./ (1 + exp(-Z1)); end
    z2 = W2 * A1 + b2;
    if strcmp(act_o, 'sig'), o = 1 ./ (1 + exp(-z2)); else, o = z2; end
    o_col = o';
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
