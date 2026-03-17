%% 1. 数据加载与预处理
clc; clear; close all;

% 假设数据已加载为 X(250×12), Y(250×6)
% X: 12个输入特征，Y: [F_peak, t_peak, F_plat_start, t_plat_start, t_plat_end, t_end]
%load('your_data.mat');

load('X.mat');
load('Y.mat');

% 输入输出标准化
[X_normalized, X_mean, X_std] = zscore(X);
[Y_normalized, Y_mean, Y_std] = zscore(Y);

% 添加高斯噪声增强数据
noise_level = 0.05;
X_aug = [X_normalized; X_normalized + noise_level*randn(size(X_normalized))];
Y_aug = [Y_normalized; Y_normalized + noise_level*randn(size(Y_normalized))];

% 划分训练集(80%)和验证集(20%)
rng(42); % 固定随机种子
cv = cvpartition(size(X_aug,1), 'HoldOut', 0.2);
X_train = X_aug(cv.training,:);
Y_train = Y_aug(cv.training,:);
X_val = X_aug(cv.test,:);
Y_val = Y_aug(cv.test,:);

%% 2. 定义物理约束的PINN模型
inputSize = 12;
outputSize = 6;

% 网络结构
layers = [
    featureInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    fullyConnectedLayer(64, 'Name', 'fc1')
    layerNormalizationLayer('Name', 'ln1')
    swishLayer('Name', 'swish1') % 比ReLU更平滑
    dropoutLayer(0.4, 'Name', 'drop1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    layerNormalizationLayer('Name', 'ln2')
    swishLayer('Name', 'swish2')
    dropoutLayer(0.4, 'Name', 'drop2')
    fullyConnectedLayer(outputSize, 'Name', 'output')
    ];

dlnet = dlnetwork(layers);

%% 4. 训练参数设置
% 超参数
lambda_phys = 0.8;   % 物理约束权重
lambda_krig = 0.3;   % 克里金正则化权重
numEpochs = 1000;
batchSize = 32;

%优化器设置
initialLearnRate = 0.001;
decayRate = 0.1;
decayStep = 100;
gradDecay = 0.9;
sqGradDecay = 0.999;

% 初始化Adam优化器状态
avgGrad = [];
avgSqGrad = [];

% 初始变异函数拟合
initialPred = predict(dlnet, dlarray(X_train', 'CB'));
initialResidual = extractdata(initialPred)' - Y_train;

variogram = cell(6,1); % 存储每个输出的变异函数
for i = 1:6
    [h_exp, gamma_exp] = experimentalVariogram(X_train, initialResidual(:,i));
    variogram{i} = fitVariogramModel(h_exp, gamma_exp);
end

%% 5. 训练循环
lossHistory = [];
for epoch = 1:numEpochs

    % 学习率衰减
    learnRate = initialLearnRate * decayRate^(floor(epoch/decayStep));

    % 打乱数据
    idx = randperm(size(X_train,1));
    X_train = X_train(idx,:);
    Y_train = Y_train(idx,:);

    % 小批量训练
    for i = 1:ceil(size(X_train,1)/batchSize)
        idx = (i-1)*batchSize+1 : min(i*batchSize, size(X_train,1));
        X_batch = dlarray(X_train(idx,:)', 'CB');
        Y_batch = Y_train(idx,:);

        [gradients, loss] = dlfeval(@modelGradients, dlnet, X_batch, Y_batch,...
            variogram, lambda_phys, lambda_krig);
        % 使用adamupdate更新参数
        [dlnet.Learnables, avgGrad, avgSqGrad] = adamupdate(...
            dlnet.Learnables, gradients, avgGrad, avgSqGrad, epoch, ...
            learnRate, gradDecay, sqGradDecay);

        lossHistory = [lossHistory, loss];
    end

    % 每50轮更新变异函数并验证
    if mod(epoch,50) == 0
        % 更新变异函数
        currentPred = predict(dlnet, dlarray(X_train', 'CB'));
        currentResidual = extractdata(currentPred)' - Y_train;

        % 为每个输出更新变异函数
        for i = 1:6
            [h_exp, gamma_exp] = experimentalVariogram(X_train, currentResidual(:,i));
            variogram{i} = fitVariogramModel(h_exp, gamma_exp);
        end
    end

    % 验证集评估
    Y_val_pred = predict(dlnet, dlarray(X_val', 'CB'));
    val_loss = mse(Y_val_pred, Y_val');
    fprintf('Epoch %d, Train Loss: %.4f, Val Loss: %.4f\n', epoch, loss, val_loss);
end


%% 6. 克里金残差建模与预测
% 获取神经网络预测趋势
Y_trend = predict(dlnet, dlarray(X_normalized', 'CB'));
Y_trend = extractdata(Y_trend)';

% 计算残差
residual = Y_normalized - Y_trend;

% 为每个输出变量单独训练克里金模型
krigingModels = cell(6,1);
for i = 1:6
    % 步骤1: 计算经验半变异函数
    [h_exp, gamma_exp] = experimentalVariogram(X_normalized, residual(:,i));
    
    % 步骤2: 拟合理论变异函数模型（关键修正：存储到元胞数组）
    variogram{i} = fitVariogramModel(h_exp, gamma_exp);
    
    % 步骤3: 定义预测函数（闭包捕获当前变量参数）
    krigingModels{i} = @(X_new) predictKriging(X_new, X_normalized, residual(:,i), variogram{i});
end

%% 7. 验证与可视化
figure;
if ~isempty(lossHistory)
    semilogy(double(extractdata(lossHistory)), 'LineWidth',1.5);
else
    error('lossHistory 未初始化或为空');
end
xlabel('Iteration');
ylabel('Log Loss');
title('Training Loss Progression');
grid on;

% 物理约束检查
Y_test_pred = combinedPredict(X_val, Y_mean, Y_std, dlnet, krigingModels);
check_physics_constraints(Y_test_pred);

Y_test = Y_val .* Y_std + Y_mean;
%% 预测效果可视化
figure('Position', [100 100 1400 900], 'Color','w');

% 定义输出变量名称（根据你的数据实际含义修改）
outputNames = {'F_{peak}', 't_{peak}', 'F_{plat\_start}', ...
               't_{plat\_start}', 't_{plat\_end}', 't_{end}'};

% 创建颜色映射
colors = lines(6);

for i = 1:6
    subplot(2,3,i);
    
    % 获取真实值和预测值
    y_true = Y_test(:,i);
    y_pred = Y_test_pred(:,i);
    
    % 绘制散点图
    hScatter = scatter(y_true, y_pred, 40, 'filled', ...
                      'MarkerFaceColor', colors(i,:), ...
                      'MarkerEdgeColor', 'k', ...
                      'MarkerFaceAlpha', 0.6);
    hold on;
    
    % 绘制理想对角线
    plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], ...
        '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
    
    % 计算统计指标
    R2 = 1 - sum((y_true - y_pred).^2)/sum((y_true - mean(y_true)).^2);
    MAE = mean(abs(y_true - y_pred));
    RMSE = sqrt(mean((y_true - y_pred).^2));
    
    % 添加统计信息
    text(0.05, 0.92, sprintf('R² = %.3f', R2), ...
        'Units','normalized', 'FontSize',10, 'Color','k');
    text(0.05, 0.82, sprintf('MAE = %.3f', MAE), ...
        'Units','normalized', 'FontSize',10, 'Color','k');
    text(0.05, 0.72, sprintf('RMSE = %.3f', RMSE), ...
        'Units','normalized', 'FontSize',10, 'Color','k');
    
    % 美化图形
    title(outputNames{i}, 'FontSize',12, 'FontWeight','bold');
    xlabel('True Values', 'FontSize',10);
    ylabel('Predictions', 'FontSize',10);
    grid on;
    axis equal tight
    box on;
    
    % 添加趋势线
    coeffs = polyfit(y_true, y_pred, 1);
    x_fit = linspace(min(y_true), max(y_true), 100);
    y_fit = polyval(coeffs, x_fit);
    plot(x_fit, y_fit, 'r-', 'LineWidth', 1.5);
    
    % 设置统一坐标范围
%     xlim([min(y_true)-0.1*range(y_true), max(y_true)+0.1*range(y_true)]);
%     ylim([min(y_pred)-0.1*range(y_pred), max(y_pred)+0.1*range(y_pred)]);
end

% 添加整体标题
sgtitle('Model Prediction Performance Across Output Variables', ...
        'FontSize',14, 'FontWeight','bold', 'Color','k');

% % 残差变异函数可视化
% % 需先确保已计算残差
% if ~exist('residual', 'var')
%     error('需先计算残差: residual = Y_train - Y_trend_pred');
% end
% 
% % 使用自定义的 experimentalVariogram 函数
% figure;
% [h_exp, gamma_exp] = experimentalVariogram(X_normalized, residual(:,2));
% h_theo = linspace(0, max(h_exp), 100);
% gamma_theo = variogram{2}.model(h_theo); % 假设variogram是元胞数组
% 
% plot(h_exp, gamma_exp, 'bo', 'MarkerFaceColor','b'); hold on;
% plot(h_theo, gamma_theo, 'r-', 'LineWidth',2);
% xlabel('距离 (h)');
% ylabel('\gamma(h)');
% legend('经验变异函数', '理论拟合');
% title('残差变异函数 (t_{peak})');
% grid on;