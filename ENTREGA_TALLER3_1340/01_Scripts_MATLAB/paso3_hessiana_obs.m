%% PASO 3: MOTOR MATEMATICO DE H^-1 POR PRODUCTOS EXTERNOS (WOODBURY) Y SALIENCIAS OBS
% Implementacion exacta de las ecuaciones de sensibilidad X^{[k]} (Ecs. 11, 19-21),
% matriz Hessiana de Gauss-Newton (Ec. 12), recursion de Woodbury (Ec. 17) y saliencias
% de Optimal Brain Surgeon (Hassibi & Stork, NIPS 1993) sobre las 3 Redes Finalistas.

clear; clc; close all;

%% 1. Carga de datos particionados y modelos base entrenados en el Punto 2
base_dir  = fileparts(fileparts(mfilename('fullpath')));
if isempty(base_dir), base_dir = pwd; end
datos_dir = fullfile(base_dir, '02_Datos');
fig_dir   = fullfile(base_dir, '04_Figuras');
res_dir   = fullfile(base_dir, '05_Resultados_Parciales');

part = load(fullfile(datos_dir, 'particion_datos.mat'));
red  = load(fullfile(res_dir, 'red_base_10_60_1.mat'));

X_tr = part.X_tr; y_tr = part.y_tr;
[P, d] = size(X_tr); M = 60; n = 721;

% Mascara de los 660 pesos sinapticos (protegiendo los 61 biases b1 y b2 segun Hassibi & Stork)
mask_sinapticos = true(n, 1);
mask_sinapticos(601:660) = false; % Biases capa oculta b1
mask_sinapticos(721)     = false; % Bias capa salida b2

fprintf('=========================================================================================\n');
fprintf('  PASO 3: MOTOR MATEMATICO DE LA HESSIANA INVERSA (H^-1) Y SALIENCIAS OBS (3 REDES)\n');
fprintf('=========================================================================================\n');
fprintf('Dimensiones: P_tr = %d patrones | d = %d entradas | M = %d ocultas | n = %d pesos\n', P, d, M, n);
fprintf('Desglose de parametros: 600 (W1) + 60 (b1) + 60 (W2) + 1 (b2) = 721 (660 sinapticos)\n\n');

%% 2. Verificacion 1: Derivadas Analiticas X^{[k]} vs Diferencias Finitas Centradas
% Se comprueba sobre las 3 Redes Finalistas (top3_modelos: trainbr, trainrp, trainlm)
% mas top5_modelos(4) (Config A: tanh+sig) para cubrir las 3 familias de activacion.
fprintf('--- VERIFICACION 1: Vector de Sensibilidad X^{[k]} = dF(w, in^{[k]})/dw vs Dif. Finitas ---\n');
eps_fd = 1e-6;
patrones_chk = [1, 250, 560, 850, 1120];
verif_derivadas = struct();

for m_idx = 1:4
    mod_k = red.top5_modelos(m_idx);
    w_k   = mod_k.w_vec;
    ah    = mod_k.act_hid;
    ao    = mod_k.act_out;
    
    X_ana_all = calcular_sensibilidades_obs(w_k, X_tr(patrones_chk, :), d, M, ah, ao);
    err_rel_pat = zeros(1, length(patrones_chk));
    err_abs_pat = zeros(1, length(patrones_chk));
    
    for p_i = 1:length(patrones_chk)
        x_pat = X_tr(patrones_chk(p_i), :);
        xk_num = zeros(n, 1);
        for q = 1:n
            wp = w_k; wp(q) = wp(q) + eps_fd;
            wm = w_k; wm(q) = wm(q) - eps_fd;
            xk_num(q) = (forward_mlp_single(wp, x_pat, d, M, ah, ao) - ...
                         forward_mlp_single(wm, x_pat, d, M, ah, ao)) / (2 * eps_fd);
        end
        xk_ana = X_ana_all(:, p_i);
        err_abs_pat(p_i) = max(abs(xk_ana - xk_num));
        err_rel_pat(p_i) = norm(xk_ana - xk_num, inf) / max(norm(xk_num, inf), 1e-12);
    end
    verif_derivadas(m_idx).config_name = mod_k.config_name;
    verif_derivadas(m_idx).algo        = mod_k.algo;
    verif_derivadas(m_idx).max_err_abs = max(err_abs_pat);
    verif_derivadas(m_idx).max_err_rel = max(err_rel_pat);
    fprintf('  Mod #%d: %-30s | Max Err Abs = %.2e | Max Err Rel = %.2e\n', ...
        m_idx, mod_k.config_name, max(err_abs_pat), max(err_rel_pat));
    assert(max(err_rel_pat) < 1e-8, 'Error: Sensibilidad analitica X^{[k]} difiere de diferencias finitas');
end
fprintf('  [OK] Las 3 Redes Finalistas y las 3 familias de activacion cumplen Error Relativo < 1e-8.\n\n');

%% 3. Comparativa Estructural de H y H^-1 (Woodbury, Ec. 17) en las 3 Redes Finalistas (top3_modelos)
alpha_ref = 1e-6;
I_n = eye(n);
comparativa_3redes_H = struct();

fprintf('--- COMPARATIVA ESTRUCTURAL DE H Y H^-1 (WOODBURY, alpha = 1e-6) EN LAS 3 REDES FINALISTAS ---\n');
fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n');
fprintf('Red  | Algoritmo | Configuracion                 |  Tr(H)  | lam_max(H) | Rango | cond(H+aI) | Masa Off-Diag | Sobreest. OBD | Err Woodbury\n');
fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n');

for r = 1:3
    m_r = red.top3_modelos(r);
    w_r = m_r.w_vec;
    X_r = calcular_sensibilidades_obs(w_r, X_tr, d, M, m_r.act_hid, m_r.act_out);
    H_r = (X_r * X_r') / P;
    H_r = 0.5 * (H_r + H_r');
    ev_r = sort(eig(H_r), 'descend');
    rng_r = sum(ev_r > 1e-12);
    
    H_inv_w = woodbury_hessiana_inv(X_r, alpha_ref, false);
    H_inv_d = inv(H_r + alpha_ref * I_n);
    err_w_r = norm(H_inv_w - H_inv_d, 'fro') / norm(H_inv_d, 'fro');
    
    off_fro_r = 100 * norm(H_r - diag(diag(H_r)), 'fro') / norm(H_r, 'fro');
    off_l1_r  = 100 * sum(abs(H_r(:) - reshape(diag(diag(H_r)), [], 1))) / sum(abs(H_r(:)));
    
    curv_obd_r = diag(H_r) + alpha_ref;
    curv_obs_r = 1 ./ diag(H_inv_w);
    fac_obd_r  = mean(curv_obd_r(mask_sinapticos) ./ curv_obs_r(mask_sinapticos));
    L_obs_r    = (w_r.^2) ./ (2 * diag(H_inv_w));
    
    comparativa_3redes_H(r).red_id            = r;
    comparativa_3redes_H(r).algo              = m_r.algo;
    comparativa_3redes_H(r).config_name       = m_r.config_name;
    comparativa_3redes_H(r).w_norm            = m_r.w_norm;
    comparativa_3redes_H(r).gamma_eff         = m_r.gamma_eff;
    comparativa_3redes_H(r).traza_H           = trace(H_r);
    comparativa_3redes_H(r).lambda_max_H      = ev_r(1);
    comparativa_3redes_H(r).rango_H           = rng_r;
    comparativa_3redes_H(r).cond_H_reg        = cond(H_r + alpha_ref * I_n);
    comparativa_3redes_H(r).off_diag_fro_pct  = off_fro_r;
    comparativa_3redes_H(r).off_diag_l1_pct   = off_l1_r;
    comparativa_3redes_H(r).factor_schur_obd  = fac_obd_r;
    comparativa_3redes_H(r).err_rel_woodbury  = err_w_r;
    comparativa_3redes_H(r).L_obs_syn_sorted  = sort(L_obs_r(mask_sinapticos), 'ascend');
    
    fprintf(' #%d  | %-9s | %-29s | %7.3f |  %9.3e |  %3d  |  %9.2e |     %5.2f%%    |    %6.1fx    |   %8.2e\n', ...
        r, m_r.algo, m_r.config_name, trace(H_r), ev_r(1), rng_r, ...
        comparativa_3redes_H(r).cond_H_reg, off_fro_r, fac_obd_r, err_w_r);
    assert(err_w_r < 1e-9, 'Error: Recursion de Woodbury imprecisa en top3_modelos');
end
fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n\n');

%% 4. Verificacion 2: Sensibilidad a Alpha y Traza Paso a Paso de Woodbury en la Red Campeona (Red 1: trainbr)
w_camp  = red.w_vec;
act_hid = red.act_hid;
act_out = red.act_out;

X_mat = calcular_sensibilidades_obs(w_camp, X_tr, d, M, act_hid, act_out);
H     = (X_mat * X_mat') / P;
H     = 0.5 * (H + H');

eigs_H  = sort(eig(H), 'descend');
rango_H = sum(eigs_H > 1e-12);

fprintf('--- VERIFICACION 2: Exactitud de la Recursion de Woodbury (Ec. 17) en Red Campeona (trainbr) ---\n');
fprintf('Propiedades de H (Red 1: trainbr): Traza(H) = %.4f | lambda_max(H) = %.4e | Rango efectivo = %d / %d\n', ...
    trace(H), eigs_H(1), rango_H, n);
fprintf('-----------------------------------------------------------------------------------------------------\n');
fprintf('  Alpha    |  cond(H + a*I) | Error Relativo Fro | Residuo ||H_P^-1(H+aI)-I||_inf | Mediana(L_q^OBS)\n');
fprintf('-----------------------------------------------------------------------------------------------------\n');

alphas_grid = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3];
tabla_alpha = struct();

for a_i = 1:length(alphas_grid)
    al = alphas_grid(a_i);
    H_inv_a   = woodbury_hessiana_inv(X_mat, al);
    H_reg_a   = H + al * I_n;
    H_inv_dir = inv(H_reg_a);
    
    err_rel_f = norm(H_inv_a - H_inv_dir, 'fro') / norm(H_inv_dir, 'fro');
    res_inf   = norm(H_inv_a * H_reg_a - I_n, inf);
    cond_val  = cond(H_reg_a);
    L_obs_a   = (w_camp.^2) ./ (2 * diag(H_inv_a));
    
    tabla_alpha(a_i).alpha     = al;
    tabla_alpha(a_i).cond_H    = cond_val;
    tabla_alpha(a_i).err_rel_f = err_rel_f;
    tabla_alpha(a_i).res_inf   = res_inf;
    tabla_alpha(a_i).med_L_obs = median(L_obs_a(mask_sinapticos));
    
    fprintf('  %7.0e  |   %10.2e   |      %9.2e     |            %9.2e           |     %9.2e\n', ...
        al, cond_val, err_rel_f, res_inf, tabla_alpha(a_i).med_L_obs);
    assert(err_rel_f < 1e-8, 'Error: Discrepancia excesiva en recursion de Woodbury');
end
fprintf('-----------------------------------------------------------------------------------------------------\n\n');

[H_inv, traza_woodbury] = woodbury_hessiana_inv(X_mat, alpha_ref, true);

%% 5. Analisis Estructural: No-Diagonalidad de H y Complemento de Schur (OBS vs OBD)
H_diag_only       = diag(diag(H));
peso_off_diag_fro = norm(H - H_diag_only, 'fro') / norm(H, 'fro');
peso_off_diag_l1  = sum(abs(H(:) - H_diag_only(:))) / sum(abs(H(:)));

curv_directa_obd = diag(H) + alpha_ref;
curv_schur_obs   = 1 ./ diag(H_inv);
razon_schur      = curv_schur_obs ./ curv_directa_obd;

fprintf('--- ANALISIS ESTRUCTURAL DE LA HESSIANA EN RED CAMPEONA (trainbr) ---\n');
fprintf('Peso fuera de la diagonal (Frobenius) : ||H - diag(H)||_F / ||H||_F = %.4f (%.2f%%)\n', ...
    peso_off_diag_fro, 100 * peso_off_diag_fro);
fprintf('Peso fuera de la diagonal (Norma L1)  : sum|H_ij (i!=j)| / sum|H_ij| = %.4f (%.2f%%)\n', ...
    peso_off_diag_l1, 100 * peso_off_diag_l1);
fprintf('Razon de Schur (1/[H^-1]_qq)/(H_qq+a) : Media = %.4f | Mediana = %.4f | Min = %.2e | Max = %.4f\n', ...
    mean(razon_schur), median(razon_schur), min(razon_schur), max(razon_schur));
fprintf('  -> Interpretacion: En promedio, OBD sobreestima la curvatura efectiva de cada peso en un factor de %.1fx\n', ...
    mean(1 ./ razon_schur(mask_sinapticos)));
fprintf('     debido a que ignora la compensacion quirurgica de los pesos correlacionados.\n\n');

%% 6. Comparacion de Saliencias Iniciales (Magnitud vs OBD vs OBS) y Teorema de Poda en Un Paso
L_mag = abs(w_camp);
L_obd = 0.5 * (diag(H) + alpha_ref) .* (w_camp.^2);
L_obs = (w_camp.^2) ./ (2 * diag(H_inv));

idx_syn = find(mask_sinapticos);
idx_act = idx_syn(abs(w_camp(idx_syn)) >= 0.08);
rho_mag_obs_all = corr(L_mag(idx_syn), L_obs(idx_syn), 'Type', 'Spearman');
rho_obd_obs_all = corr(L_obd(idx_syn), L_obs(idx_syn), 'Type', 'Spearman');
rho_mag_obs_act = corr(L_mag(idx_act), L_obs(idx_act), 'Type', 'Spearman');
rho_obd_obs_act = corr(L_obd(idx_act), L_obs(idx_act), 'Type', 'Spearman');

fprintf('--- CORRELACION DE RANGOS DE SPEARMAN ENTRE CRITERIOS DE SALIENCIA (Red Campeona: trainbr) ---\n');
fprintf('En los 660 pesos sinapticos totales : rho(Mag, OBS) = %.4f | rho(OBD, OBS) = %.4f\n', rho_mag_obs_all, rho_obd_obs_all);
fprintf('En los %d pesos activos (|w|>=0.08) : rho(Mag, OBS) = %.4f | rho(OBD, OBS) = %.4f\n\n', length(idx_act), rho_mag_obs_act, rho_obd_obs_act);

E_base_tr = eval_mse_only(w_camp, X_tr, y_tr, d, M, act_hid, act_out);
umbrales_prueba = [0.00, 0.04, 0.08, 0.12, 0.18, 0.25];
tabla_teorema = struct();

fprintf('--- VERIFICACION EMPIRICA DEL TEOREMA DE PODA EN UN PASO: Delta E_real EN LA RED CAMPEONA (trainbr) ---\n');
fprintf('----------------------------------------------------------------------------------------------------------------\n');
fprintf('Subconjunto Sinaptico | q*(Mag) |   Delta E(Mag)  | q*(OBD) |   Delta E(OBD)  | q*(OBS) |   Delta E(OBS)  | Mejora OBS/OBD\n');
fprintf('----------------------------------------------------------------------------------------------------------------\n');

for u_i = 1:length(umbrales_prueba)
    thr = umbrales_prueba(u_i);
    cand = idx_syn(abs(w_camp(idx_syn)) >= thr);
    [~, im] = min(L_mag(cand)); q_m = cand(im);
    [~, id] = min(L_obd(cand)); q_d = cand(id);
    [~, is] = min(L_obs(cand)); q_s = cand(is);
    
    w_m = w_camp; w_m(q_m) = 0;
    dE_m = eval_mse_only(w_m, X_tr, y_tr, d, M, act_hid, act_out) - E_base_tr;
    
    w_d = w_camp; w_d(q_d) = 0;
    dE_d = eval_mse_only(w_d, X_tr, y_tr, d, M, act_hid, act_out) - E_base_tr;
    
    w_s = w_camp - (w_camp(q_s) / H_inv(q_s, q_s)) * H_inv(:, q_s);
    w_s(q_s) = 0;
    dE_s = eval_mse_only(w_s, X_tr, y_tr, d, M, act_hid, act_out) - E_base_tr;
    
    tabla_teorema(u_i).umbral  = thr;
    tabla_teorema(u_i).n_cand  = length(cand);
    tabla_teorema(u_i).q_mag   = q_m; tabla_teorema(u_i).dE_mag = dE_m;
    tabla_teorema(u_i).q_obd   = q_d; tabla_teorema(u_i).dE_obd = dE_d;
    tabla_teorema(u_i).q_obs   = q_s; tabla_teorema(u_i).dE_obs = dE_s;
    
    fprintf('|w_q| >= %.2f (N=%3d) |  q=%3d  | %+14.4e |  q=%3d  | %+14.4e |  q=%3d  | %+14.4e |    %6.1fx\n', ...
        thr, length(cand), q_m, dE_m, q_d, dE_d, q_s, dE_s, abs(dE_d) / max(abs(dE_s), 1e-18));
end
fprintf('----------------------------------------------------------------------------------------------------------------\n\n');

%% 7. Guardado de Resultados Parciales del Punto 3
save(fullfile(res_dir, 'hessiana_obs_punto3.mat'), ...
    'H', 'H_inv', 'alpha_ref', 'X_mat', 'eigs_H', 'rango_H', ...
    'mask_sinapticos', 'L_mag', 'L_obd', 'L_obs', ...
    'curv_directa_obd', 'curv_schur_obs', 'razon_schur', ...
    'peso_off_diag_fro', 'peso_off_diag_l1', ...
    'verif_derivadas', 'comparativa_3redes_H', 'tabla_alpha', 'traza_woodbury', 'tabla_teorema');

%% 8. FIGURA 7: Mapa de Calor de H y H^-1, Espectro y Convergencia de Woodbury (Red Campeona: trainbr)
fig7 = figure('Name', 'Estructura de H, H^-1 y Recursion de Woodbury', 'Position', [60, 60, 1340, 740], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Mapa de calor de log10(|H|) mostrando estructura fuertemente no-diagonal
nexttile;
imagesc(log10(abs(H) + 1e-12));
colormap(gca, turbo); cb1 = colorbar; cb1.Label.String = 'log_{10}(|H_{ij}|)';
clim([-6.5, 0.5]); axis square; hold on;
xline(600.5, 'w--', 'LineWidth', 1.2); xline(660.5, 'w--', 'LineWidth', 1.2);
yline(600.5, 'w--', 'LineWidth', 1.2); yline(660.5, 'w--', 'LineWidth', 1.2);
xlabel('Índice de Parámetro q (1..600: W_1, 601..660: b_1, 661..721: W_2, b_2)', 'FontSize', 9.5);
ylabel('Índice de Parámetro p', 'FontSize', 9.5);
title(sprintf('Matriz Hessiana |H| en Red 1 (trainbr, Off-Diag = %.2f%%)', 100 * peso_off_diag_fro), ...
    'FontSize', 11, 'FontWeight', 'bold');
hold off;

% Panel 2: Mapa de calor de log10(|H^-1|) obtenida por Recursion de Woodbury (alpha = 1e-6)
nexttile;
imagesc(log10(abs(H_inv) + 1e-12));
colormap(gca, turbo); cb2 = colorbar; cb2.Label.String = 'log_{10}(|[H^{-1}]_{ij}|)';
clim([0, 6]); axis square; hold on;
xline(600.5, 'w--', 'LineWidth', 1.2); xline(660.5, 'w--', 'LineWidth', 1.2);
yline(600.5, 'w--', 'LineWidth', 1.2); yline(660.5, 'w--', 'LineWidth', 1.2);
xlabel('Índice de Parámetro q', 'FontSize', 9.5);
ylabel('Índice de Parámetro p', 'FontSize', 9.5);
title(sprintf('Inversa de Woodbury |H^{-1}| (\\alpha = 10^{-6}, Err Rel = %.2e)', tabla_alpha(2).err_rel_f), ...
    'FontSize', 11, 'FontWeight', 'bold');
hold off;

% Panel 3: Evolucion de la recursion de Woodbury paso a paso m = 1..1120
nexttile;
yyaxis left;
plot(traza_woodbury.pasos_m, traza_woodbury.traza_H_inv, '-', 'LineWidth', 2.0, 'Color', [0.0000 0.4470 0.7410]);
ylabel('Traza(H_m^{-1}) / n (Varianza Paramétrica Media)', 'FontSize', 10);
set(gca, 'YScale', 'log');
yyaxis right;
semilogy(traza_woodbury.pasos_m, traza_woodbury.err_rel_paso, '--o', 'LineWidth', 1.8, 'MarkerSize', 4, ...
    'Color', [0.8500 0.3250 0.0980], 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
ylabel('Error Relativo ||H_m^{-1} - (H_m + \alpha I)^{-1}||_F / ||\cdot||_F', 'FontSize', 10);
ylim([1e-16, 1e-8]);
grid on; xlabel('Número de Patrones Incorporados m (1 \leq m \leq P_{tr} = 1120)', 'FontSize', 10);
title('Estabilidad Numérica de la Recursión de Woodbury (Ec. 17)', 'FontSize', 11, 'FontWeight', 'bold');
legend({'Traza(H_m^{-1})/n', 'Error Relativo vs Inversión Directa'}, 'Location', 'northeast', 'FontSize', 9);

% Panel 4: Espectro de Autovalores de H en las 3 Redes Finalistas (trainbr, trainrp, trainlm)
nexttile; hold on;
cols_3r = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.4940 0.1840 0.5560];
styles_3r = {'-', '--', '-.'};
for r = 1:3
    m_r = red.top3_modelos(r);
    X_r = calcular_sensibilidades_obs(m_r.w_vec, X_tr, d, M, m_r.act_hid, m_r.act_out);
    H_r = 0.5 * ((X_r * X_r') / P + ((X_r * X_r') / P)');
    ev_r = sort(eig(H_r), 'descend');
    semilogy(1:n, max(ev_r, 1e-16), styles_3r{r}, 'LineWidth', 1.9, 'Color', cols_3r(r,:), ...
        'DisplayName', sprintf('Red %d: %s (Rango=%d)', r, m_r.algo, comparativa_3redes_H(r).rango_H));
end
yline(1e-6, 'g-.', 'LineWidth', 1.4, 'DisplayName', '\alpha = 10^{-6} (Referencia OBS)');
set(gca, 'YScale', 'log'); grid on; xlim([1 n]); ylim([1e-16, 20]);
xlabel('Índice de Autovalor Ordenado (i = 1, ..., 721)', 'FontSize', 10);
ylabel('Autovalor \lambda_i(H) (escala log)', 'FontSize', 10);
title('Espectro de Autovalores de H en las 3 Redes Finalistas', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 8.5);
hold off;

exportgraphics(fig7, fullfile(fig_dir, 'fig7_mapa_hessiana_y_woodbury.png'), 'Resolution', 200);

%% 9. FIGURA 8: Complemento de Schur, Saliencias OBS vs OBD vs Magnitud y Teorema de Poda
fig8 = figure('Name', 'Comparativa Saliencias OBS vs OBD vs Magnitud', 'Position', [60, 60, 1340, 740], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Curvatura Directa H_qq (OBD) vs Curvatura Efectiva de Schur 1/[H^-1]_qq (OBS)
nexttile;
loglog(curv_directa_obd(idx_act), curv_schur_obs(idx_act), 'o', 'MarkerSize', 5, ...
    'MarkerFaceColor', [0.0000 0.4470 0.7410], 'MarkerEdgeColor', 'none'); hold on;
lims_c = [5e-7, 1e0];
loglog(lims_c, lims_c, 'k--', 'LineWidth', 1.5);
grid on; xlim(lims_c); ylim(lims_c); axis square;
xlabel('Curvatura Diagonal Directa H_{qq} + \alpha (Supuesto OBD)', 'FontSize', 10);
ylabel('Curvatura Efectiva de Schur 1 / [H^{-1}]_{qq} (OBS)', 'FontSize', 10);
title(sprintf('Reducción de Curvatura por Complemento de Schur (Sobreest. OBD = %.0fx)', comparativa_3redes_H(1).factor_schur_obd), ...
    'FontSize', 10.5, 'FontWeight', 'bold');
legend({'Pesos Sinápticos Activos (Red 1: trainbr)', 'Línea 1:1 (H Diagonal: OBS = OBD)'}, 'Location', 'northwest', 'FontSize', 8.5);
hold off;

% Panel 2: Saliencias ordenadas L_q^OBS en las 3 Redes Finalistas + OBD y Magnitud en Red 1
nexttile;
semilogy(1:length(idx_syn), comparativa_3redes_H(1).L_obs_syn_sorted, '-', 'LineWidth', 2.1, 'Color', [0.0000 0.4470 0.7410]); hold on;
semilogy(1:length(idx_syn), comparativa_3redes_H(2).L_obs_syn_sorted, ':', 'LineWidth', 1.8, 'Color', [0.0000 0.6500 0.7500]);
semilogy(1:length(idx_syn), comparativa_3redes_H(3).L_obs_syn_sorted, '--', 'LineWidth', 1.6, 'Color', [0.4940 0.1840 0.5560]);
semilogy(1:length(idx_syn), sort(L_obd(idx_syn), 'ascend'), '-.', 'LineWidth', 1.7, 'Color', [0.8500 0.3250 0.0980]);
semilogy(1:length(idx_syn), sort(0.5 * (w_camp(idx_syn).^2), 'ascend'), '--', 'LineWidth', 1.5, 'Color', [0.4660 0.6740 0.1880]);
grid on; xlim([1 length(idx_syn)]); ylim([1e-12, 1e1]);
xlabel('Orden de Poda de los Pesos Sinápticos (1 a 660)', 'FontSize', 10);
ylabel('Saliencia Estimada L_q (escala log)', 'FontSize', 10);
title('Perfil Ordenado de Saliencias: 3 Redes Finalistas y OBS vs OBD/Mag', 'FontSize', 10.5, 'FontWeight', 'bold');
legend({'L_q^{OBS} Red 1 (trainbr)', 'L_q^{OBS} Red 2 (trainrp)', 'L_q^{OBS} Red 3 (trainlm)', ...
        'L_q^{OBD} Red 1 (Diagonal)', '(1/2)w_q^2 Red 1 (Magnitud)'}, 'Location', 'northwest', 'FontSize', 8.0);
hold off;

% Panel 3: Discrepancia de Ranking entre Magnitud, OBD y OBS en los pesos activos
nexttile;
[~, rank_obs_act] = sort(L_obs(idx_act), 'ascend');
[~, pos_obs] = sort(rank_obs_act);
[~, rank_obd_act] = sort(L_obd(idx_act), 'ascend');
[~, pos_obd] = sort(rank_obd_act);
[~, rank_mag_act] = sort(L_mag(idx_act), 'ascend');
[~, pos_mag] = sort(rank_mag_act);
scatter(pos_obs, pos_mag, 28, [0.4660 0.6740 0.1880], 'filled', 'MarkerFaceAlpha', 0.65); hold on;
scatter(pos_obs, pos_obd, 28, [0.8500 0.3250 0.0980], 'd', 'filled', 'MarkerFaceAlpha', 0.75);
plot([1 length(idx_act)], [1 length(idx_act)], 'k--', 'LineWidth', 1.3);
grid on; xlim([1 length(idx_act)]); ylim([1 length(idx_act)]); axis square;
xlabel('Prioridad de Poda según OBS (1 = Podar Primero)', 'FontSize', 10);
ylabel('Prioridad de Poda según Magnitud / OBD', 'FontSize', 10);
title(sprintf('Reordenamiento de Prioridad en Pesos Activos (\\rho_{Mag}=%.2f, \\rho_{OBD}=%.2f)', ...
    rho_mag_obs_act, rho_obd_obs_act), 'FontSize', 10.5, 'FontWeight', 'bold');
legend({'Ranking Magnitud vs OBS', 'Ranking OBD vs OBS', 'Concordancia Exacta'}, 'Location', 'northwest', 'FontSize', 8.5);
hold off;

% Panel 4: Verificacion empirica del incremento real del error Delta E_real al podar 1 peso activo
nexttile;
idx_sub_plot = 2:length(umbrales_prueba); % Umbrales |w| >= 0.04 a 0.25
mat_dE = zeros(length(idx_sub_plot), 3);
lbl_u  = cell(length(idx_sub_plot), 1);
for ii = 1:length(idx_sub_plot)
    t_row = tabla_teorema(idx_sub_plot(ii));
    mat_dE(ii, :) = [abs(t_row.dE_mag), abs(t_row.dE_obd), abs(t_row.dE_obs)];
    lbl_u{ii} = sprintf('|w|\\geq%.2f', t_row.umbral);
end
b_teor = bar(1:length(idx_sub_plot), mat_dE, 'grouped');
b_teor(1).FaceColor = [0.4660 0.6740 0.1880];
b_teor(2).FaceColor = [0.8500 0.3250 0.0980];
b_teor(3).FaceColor = [0.0000 0.4470 0.7410];
set(gca, 'YScale', 'log'); grid on;
xticks(1:length(idx_sub_plot)); xticklabels(lbl_u);
ylim([1e-11, 1e-1]);
xlabel('Régimen de Magnitud del Peso Podado', 'FontSize', 10);
ylabel('Incremento Real del Error |\Delta E_{real}| (log)', 'FontSize', 10);
title('Verificación Empírica: \Delta E(Magnitud) \geq \Delta E(OBD) >> \Delta E(OBS)', 'FontSize', 10.5, 'FontWeight', 'bold');
legend({'Magnitud (sin cirugía)', 'OBD (sin cirugía)', 'OBS (con cirugía H^{-1})'}, 'Location', 'northwest', 'FontSize', 8.5);

exportgraphics(fig8, fullfile(fig_dir, 'fig8_comparativa_saliencias_obs_obd_mag.png'), 'Resolution', 200);
fprintf('Figuras fig7 y fig8 y archivo hessiana_obs_punto3.mat generados exitosamente.\n');

%% =========================================================================
%% FUNCIONES MATEMATICAS DEL MOTOR OBS (DISCIPLINA PONYTAIL)
%% =========================================================================

function X_mat = calcular_sensibilidades_obs(w, X, d, M, act_h, act_o, mask_activos)
    % Calcula la matriz X_mat (n x P) cuyas columnas son X^{[k]} = dF(w, in^{[k]})/dw
    % segun Hassibi & Stork (1993), Ecs. 11 y 19-21.
    if nargin > 6 && ~isempty(mask_activos)
        w = w .* mask_activos;
    end
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
        do = ones(1, P);
    end
    
    delta1 = (W2' * do) .* dA1; % 60 x P
    X_W1  = repmat(delta1, d, 1) .* repelem(X', M, 1); % 600 x P
    X_b1  = delta1;                                     % 60 x P
    X_W2  = A1 .* do;                                   % 60 x P
    X_b2  = do;                                         % 1 x P
    X_mat = [X_W1; X_b1; X_W2; X_b2];
    
    if nargin > 6 && ~isempty(mask_activos)
        X_mat = X_mat .* mask_activos;
    end
end

function [H_inv, traza_info] = woodbury_hessiana_inv(X_mat, alpha, guardar_traza)
    % Implementa la recursion de Sherman-Morrison-Woodbury (Ec. 17 de Hassibi & Stork):
    % H_{m+1}^{-1} = H_m^{-1} - (H_m^{-1} X^{[m+1]} X^{[m+1]T} H_m^{-1}) / (P + X^{[m+1]T} H_m^{-1} X^{[m+1]})
    if nargin < 3, guardar_traza = false; end
    [n, P] = size(X_mat);
    H_inv  = (1 / alpha) * eye(n);
    
    if guardar_traza
        pasos_m = unique([1, 10, 25, 50, 100:100:P, P]);
        traza_H_inv  = zeros(size(pasos_m));
        err_rel_paso = zeros(size(pasos_m));
        H_acum       = zeros(n, n);
        idx_t        = 1;
    end
    
    for m = 1:P
        xk = X_mat(:, m);
        z  = H_inv * xk;
        H_inv = H_inv - (z * z') / (P + xk' * z);
        
        if guardar_traza
            H_acum = H_acum + (xk * xk') / P;
            if m == pasos_m(idx_t)
                H_inv_sym = 0.5 * (H_inv + H_inv');
                H_dir_m   = inv(H_acum + alpha * eye(n));
                traza_H_inv(idx_t)  = trace(H_inv_sym) / n;
                err_rel_paso(idx_t) = norm(H_inv_sym - H_dir_m, 'fro') / norm(H_dir_m, 'fro');
                idx_t = idx_t + 1;
            end
        end
    end
    H_inv = 0.5 * (H_inv + H_inv');
    if guardar_traza
        traza_info.pasos_m      = pasos_m;
        traza_info.traza_H_inv  = traza_H_inv;
        traza_info.err_rel_paso = err_rel_paso;
    else
        traza_info = struct();
    end
end

function o = forward_mlp_single(w, x_row, d, M, act_h, act_o)
    W1 = reshape(w(1:M*d), M, d); b1 = w(M*d+1 : M*(d+1));
    W2 = reshape(w(M*(d+1)+1 : M*(d+2)), 1, M); b2 = w(end);
    z1 = W1 * x_row' + b1;
    if strcmp(act_h, 'tanh'), a1 = tanh(z1); else, a1 = 1 ./ (1 + exp(-z1)); end
    z2 = W2 * a1 + b2;
    if strcmp(act_o, 'sig'), o = 1 ./ (1 + exp(-z2)); else, o = z2; end
end

function E_mse = eval_mse_only(w, X, y, d, M, act_h, act_o)
    W1 = reshape(w(1:M*d), M, d); b1 = w(M*d+1 : M*(d+1));
    W2 = reshape(w(M*(d+1)+1 : M*(d+2)), 1, M); b2 = w(end);
    Z1 = W1 * X' + b1;
    if strcmp(act_h, 'tanh'), A1 = tanh(Z1); else, A1 = 1 ./ (1 + exp(-Z1)); end
    z2 = W2 * A1 + b2;
    if strcmp(act_o, 'sig'), o = 1 ./ (1 + exp(-z2)); else, o = z2; end
    E_mse = 0.5 * mean((y - o').^2);
end
