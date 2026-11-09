function traj = ejecutar_poda_obs_pura(w_init, X_tr, y_tr, X_val, y_val, X_te, y_te, X_rep, d, M, act_h, act_o, mask_podables, max_podas, alpha)
% EJECUTAR_PODA_OBS_PURA Poda Optimal Brain Surgeon PURA (Hassibi & Stork, 1993)
% CERO REENTRENAMIENTO: Aplica unicamente la cirugia analitica dw de segundo orden
% y la inversion recursiva de la matriz Hessiana via Woodbury.

    P = size(X_tr, 1); n = length(w_init);
    w = w_init;
    mask_act = true(n, 1);
    
    E_tr = zeros(max_podas+1, 1); err_tr = zeros(max_podas+1, 1);
    E_val= zeros(max_podas+1, 1); err_val= zeros(max_podas+1, 1);
    E_te = zeros(max_podas+1, 1); err_te = zeros(max_podas+1, 1);
    w_hist = zeros(n, max_podas+1);
    historial_q = zeros(max_podas, 1);
    
    [E_tr(1), err_tr(1)]   = eval_red(w, X_tr, y_tr, d, M, act_h, act_o);
    [E_val(1), err_val(1)] = eval_red(w, X_val, y_val, d, M, act_h, act_o);
    [E_te(1), err_te(1)]   = eval_red(w, X_te, y_te, d, M, act_h, act_o);
    w_hist(:, 1) = w;
    
    X_m   = calc_sens(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
    H_inv = woodbury_hessiana_inv(X_m, alpha);
    
    for step = 1:max_podas
        rem_syn = sum(mask_act & mask_podables);
        
        % Sincronizacion periodica de H_inv mediante Woodbury en el subespacio activo (sin optimizacion)
        if rem_syn <= 180 && mod(step, 5) == 0
            act_idx = find(mask_act);
            X_m = calc_sens(w, X_tr, y_tr, X_rep, d, M, act_h, act_o, mask_act);
            H_inv = zeros(n, n);
            H_inv(act_idx, act_idx) = woodbury_hessiana_inv(X_m(act_idx, :), alpha);
        end
        
        diag_Hinv = max(diag(H_inv), 1e-15);
        L = (w.^2) ./ (2 * diag_Hinv);
        
        % Candado 1: L_q = +Inf para pesos inactivos y sesgos
        L(~(mask_act & mask_podables)) = inf;
        [~, q_star] = min(L);
        historial_q(step) = q_star;
        
        % Ajuste analitico OBS puro (Ec. 5 Hassibi & Stork)
        col_q = H_inv(:, q_star);
        h_qq  = H_inv(q_star, q_star);
        w = w - (w(q_star) / h_qq) * col_q;
        H_inv = H_inv - (col_q * col_q') / h_qq;
        
        % Candados 2 y 3: fijar en cero exacto y aplicar mascara
        mask_act(q_star) = false;
        w(q_star) = 0;
        w = w .* double(mask_act);
        
        [E_tr(step+1), err_tr(step+1)]   = eval_red(w, X_tr, y_tr, d, M, act_h, act_o);
        [E_val(step+1), err_val(step+1)] = eval_red(w, X_val, y_val, d, M, act_h, act_o);
        [E_te(step+1), err_te(step+1)]   = eval_red(w, X_te, y_te, d, M, act_h, act_o);
        w_hist(:, step+1) = w;
    end
    
    traj.E_tr = E_tr; traj.err_tr = err_tr;
    traj.E_val= E_val; traj.err_val= err_val;
    traj.E_te = E_te; traj.err_te = err_te;
    traj.w_hist = w_hist;
    traj.historial_q = historial_q;
end

function [E, err] = eval_red(w, X, y, d, M, act_h, act_o)
    W1 = reshape(w(1:M*d), M, d); b1 = w(M*d+1:M*d+M);
    W2 = reshape(w(M*d+M+1:M*d+M+M), 1, M); b2 = w(end);
    Z1 = X * W1' + b1';
    if strcmp(act_h, 'tanh'), A1 = tanh(Z1); else, A1 = 1 ./ (1 + exp(-Z1)); end
    Z2 = A1 * W2' + b2;
    if strcmp(act_o, 'linear') || strcmp(act_o, 'lin')
        o = Z2;
    else
        o = 1 ./ (1 + exp(-Z2));
    end
    y_pred = double(o >= 0.5);
    E = mean((y - o).^2) / 2;
    err = mean(y ~= y_pred) * 100;
end

function X_mat = calc_sens(w, X, y, X_rep, d, M, act_h, act_o, mask_act)
    P = size(X, 1); n = length(w);
    W1 = reshape(w(1:M*d), M, d); b1 = w(M*d+1:M*d+M);
    W2 = reshape(w(M*d+M+1:M*d+M+M), 1, M); b2 = w(end);
    Z1 = X * W1' + b1';
    if strcmp(act_h, 'tanh'), A1 = tanh(Z1); d_hid = 1 - A1.^2; else, A1 = 1 ./ (1 + exp(-Z1)); d_hid = A1 .* (1 - A1); end
    Z2 = A1 * W2' + b2;
    if strcmp(act_o, 'linear') || strcmp(act_o, 'lin'), d_out = ones(P, 1); else, o = 1 ./ (1 + exp(-Z2)); d_out = o .* (1 - o); end
    d1 = (d_out * W2) .* d_hid;
    X_mat = zeros(n, P);
    for p = 1:P
        xp = X(p, :)'; d1_p = d1(p, :)'; a1_p = A1(p, :)';
        X_mat(1:600, p)   = reshape(d1_p * xp', [600, 1]);
        X_mat(601:660, p) = d1_p;
        X_mat(661:720, p) = d_out(p) * a1_p;
        X_mat(721, p)     = d_out(p);
    end
    X_mat(~mask_act, :) = 0;
end
