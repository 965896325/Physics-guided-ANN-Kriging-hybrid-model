function Y_pred = combinedPredict(X_new, Y_mean, Y_std, dlnet, krigingModels)

% 神经网络预测趋势项
Y_trend_norm = predict(dlnet, dlarray(X_new', 'CB'));
Y_trend_norm = double(extractdata(Y_trend_norm));
Y_trend = Y_trend_norm' .* Y_std + Y_mean; % 反标准化趋势项

% 克里金残差修正
residual_pred_norm = zeros(size(X_new,1),6);
residual_pred = zeros(size(X_new,1),6);
for i = 1:6
    residual_pred_norm(:,i) = krigingModels{i}(X_new);
    residual_pred(:,i) = residual_pred_norm(:,i) * Y_std(i); % 反标准化
end

% 最终预测值
Y_pred = Y_trend + residual_pred;

% 检查负值比例
neg_mask = Y_pred < 0;
fprintf('负值比例: \n');
disp([sum(neg_mask(:,1))/size(Y_pred,1), ... % F_peak
    sum(neg_mask(:,2))/size(Y_pred,1), ... % t_peak
    sum(neg_mask(:,4))/size(Y_pred,1)]);   % t_plat_start
end