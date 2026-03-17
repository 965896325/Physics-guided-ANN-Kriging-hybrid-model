function [gradients, totalLoss] = modelGradients(dlnet, X_batch, Y_batch, variogram, lambda_phys, lambda_krig)
    Y_pred = forward(dlnet, X_batch);
    loss_data = mse(Y_pred, Y_batch');
    loss_phys = physics_constraints(Y_pred);
    
    residual = extractdata(Y_pred)' - Y_batch;
    batch_size = size(residual,1);
    loss_krig = 0;
    
    for i = 1:6
        residual_i = residual(:,i);
        X_batch_numeric = extractdata(X_batch)';
        
        D = pdist2(X_batch_numeric, X_batch_numeric);
        Sigma = variogram{i}.sill * exp(-3*D/variogram{i}.range) + ...
               variogram{i}.nugget*eye(batch_size) + 1e-6*eye(batch_size);
        
        % 使用Cholesky分解
        [L, flag] = chol(Sigma, 'lower');
        if flag == 0
            z = L \ residual_i;
            loss_term = z' * z;
        else
            % 增强正则化并重新尝试
            Sigma = Sigma + 1e-5*eye(batch_size);
            L = chol(Sigma, 'lower');
            z = L \ residual_i;
            loss_term = z' * z;
        end
        loss_krig = loss_krig + loss_term;
    end
    loss_krig = loss_krig / batch_size;
    
    totalLoss = loss_data + lambda_phys*loss_phys + lambda_krig*loss_krig;
    gradients = dlgradient(totalLoss, dlnet.Learnables);
end
