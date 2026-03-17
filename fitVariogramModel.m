function variogram = fitVariogramModel(h_exp, gamma_exp)
    % 数据有效性检查
    valid_idx = ~isnan(gamma_exp) & (h_exp > 0);
    h_exp = h_exp(valid_idx);
    gamma_exp = gamma_exp(valid_idx);
    
    if isempty(h_exp)
        variogram = struct('nugget',0, 'sill',1, 'range',1, 'model',@(h)0);
        return;
    end

    % 鲁棒的初始参数
    max_gamma = max(gamma_exp);
    max_h = max(h_exp);
    initialParams = [
        max(0.1*max_gamma, 1e-3),  
        max(max_gamma*1.1, 1e-3),   
        max(0.5*max_h, 1e-3)
    ];

    % 带约束的模型函数
    model = @(params, h) params(1) + ...
        max(params(2)-params(1),1e-3) * (1 - exp(-3*max(h,1e-3)/max(params(3),1e-3)));
    
    costFunc = @(params) sum((model(params, h_exp) - gamma_exp).^2);

    % 优化约束
    A = [1, -1, 0; 0, 0, -1];
    b = [1e-3; 0];
    lb = [0, 0, 1e-3];
    ub = [Inf, Inf, Inf];
    
    options = optimoptions('fmincon', 'Display','notify',...
        'MaxFunctionEvaluations',1e4, 'FunctionTolerance',1e-6);
    
    params = fmincon(costFunc, initialParams, A, b, [], [], lb, ub, [], options);
    
    % 结果处理
    variogram.nugget = max(params(1), 0);
    variogram.sill = max(params(2), variogram.nugget + 1e-3);
    variogram.range = max(params(3), 1e-3);
    variogram.model = @(h) variogram.nugget + ...
        (variogram.sill - variogram.nugget) * (1 - exp(-3*max(h,1e-3)/variogram.range));
end