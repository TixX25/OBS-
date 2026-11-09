function [H_inv, traza_info] = woodbury_hessiana_inv(X_mat, alpha, guardar_traza)
% WOODBURY_HESSIANA_INV Inversion recursiva de la matriz Hessiana aproximada
% por productos externos mediante la identidad de Sherman-Morrison-Woodbury.
%
% Referencia:
%   Hassibi, B. & Stork, D. G. (1993). Second order derivatives for network
%   pruning: Optimal Brain Surgeon. NIPS 5, pp. 164-171 (Ecuacion 17).
%
% Ecuacion recursiva:
%   H_0^{-1} = (1 / alpha) * I_n
%   H_{m+1}^{-1} = H_m^{-1} - \frac{H_m^{-1} X^{[m+1]} (X^{[m+1]})^T H_m^{-1}}
%                                  {P + (X^{[m+1]})^T H_m^{-1} X^{[m+1]}}
%
% Entradas:
%   X_mat         : Matriz de sensibilidades [n x P], donde n es el numero
%                   de pesos (o pesos activos) y P el numero de patrones.
%   alpha         : Parametro de amortiguamiento espectral (Tikhonov).
%   guardar_traza : (Opcional, defecto=false) Booleano para auditar la traza.
%
% Salidas:
%   H_inv         : Inversa estimada (H + alpha*I)^{-1} [n x n], simetrizada.
%   traza_info    : (Opcional) Estructura con metricas de estabilidad por paso.

    if nargin < 3, guardar_traza = false; end
    [n, P] = size(X_mat);
    
    % Inicializacion con la matriz de regularizacion (H_0^{-1} = alpha^{-1} * I)
    H_inv = (1 / alpha) * eye(n);
    
    if guardar_traza
        pasos_m = unique([1, 10, 25, 50, 100:100:P, P]);
        traza_H_inv  = zeros(size(pasos_m));
        err_rel_paso = zeros(size(pasos_m));
        H_acum       = zeros(n, n);
        idx_t        = 1;
    end
    
    if ~guardar_traza
        % Identidad de Woodbury en bloques vectorizados (Sherman-Morrison generalizada)
        bs = min(56, P);
        for m = 1:bs:P
            m_end = min(m + bs - 1, P);
            Xb = X_mat(:, m:m_end);
            Zb = H_inv * Xb;
            kb = size(Xb, 2);
            H_inv = H_inv - Zb * ((P * eye(kb) + Xb' * Zb) \ Zb');
        end
    else
        % Bucle recursivo estricto patron a patron (Hassibi & Stork, Ec. 17 con auditoria)
        for m = 1:P
            xk = X_mat(:, m);
            z  = H_inv * xk;
            denominador = P + xk' * z;
            H_inv = H_inv - (z * z') / denominador;
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
    
    % Simetrizacion numerica para neutralizar asimetrias por redondeo IEEE-754
    H_inv = 0.5 * (H_inv + H_inv');
    
    if guardar_traza
        traza_info.pasos_m      = pasos_m;
        traza_info.traza_H_inv  = traza_H_inv;
        traza_info.err_rel_paso = err_rel_paso;
    else
        traza_info = struct();
    end
end
