function [h, gamma] = experimentalVariogram(X, residual)
    % 计算所有点对之间的距离和残差差异
    n = size(X,1);
    D = pdist2(X, X); % 距离矩阵
    R = pdist2(residual, residual); % 残差差异矩阵
    pairs = nchoosek(1:n,2); % 所有点对
    
    % 提取距离和半方差
    h_all = D(sub2ind(size(D), pairs(:,1), pairs(:,2)));
    gamma_all = 0.5 * R(sub2ind(size(R), pairs(:,1), pairs(:,2))).^2;
    
    % 分箱统计
    max_h = max(h_all);
    num_bins = 20;
    bin_edges = linspace(0, max_h, num_bins+1);
    h = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
    gamma = zeros(size(h));
    
    for i = 1:num_bins
        idx = (h_all >= bin_edges(i)) & (h_all < bin_edges(i+1));
        gamma(i) = mean(gamma_all(idx));
    end
end