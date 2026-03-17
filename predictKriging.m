function pred = predictKriging(X_new, X_train, residual_train, variogram)
    % 输入:
    % X_new: [m × 12] 新样本输入（已标准化）
    % X_train: [n × 12] 训练输入（已标准化）
    % residual_train: [n × 1] 当前输出变量的训练残差
    % variogram: 结构体包含拟合的变异函数参数
    
    % 计算训练集协方差矩阵
    D_train = pdist2(X_train, X_train);
    Sigma_train = variogram.nugget * eye(size(D_train)) + ...
                 (variogram.sill - variogram.nugget) * exp(-3*D_train/variogram.range);
    
    % 计算新点与训练点的协方差
    D_new = pdist2(X_new, X_train);
    Sigma_new = variogram.nugget * ones(size(D_new)) + ...
               (variogram.sill - variogram.nugget) * exp(-3*D_new/variogram.range);
    
    % 克里金权重计算（避免直接求逆）
    weights = Sigma_new / Sigma_train; % 等价于 Sigma_new * inv(Sigma_train)
    
    % 预测残差
    pred = weights * residual_train;
end