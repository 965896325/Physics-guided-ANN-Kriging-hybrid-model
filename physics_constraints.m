function loss = physics_constraints(Y_pred)
    % 物理约束函数
    % 时间顺序约束: t_peak < t_plat_start < t_plat_end < t_end
    alpha = 0.1; % 安全间隔
    t_peak = Y_pred(2,:);
    t_plat_start = Y_pred(4,:);
    t_plat_end = Y_pred(5,:);
    t_end = Y_pred(6,:);
    
    loss_time = sum(...
        relu(t_peak - t_plat_start) + ...
        relu(t_plat_start - t_plat_end) + ...
        relu(t_plat_end - t_end)) + alpha;
    
    % 力大小约束: F_plat_start < F_peak
    F_peak = Y_pred(1,:);
    F_plat_start = Y_pred(3,:);
    loss_force = sum(relu(F_plat_start - F_peak)) + alpha;
    
    % 非负约束
    non_neg_vars = Y_pred(:,[1,2,4,5,6]); % F_peak, t_peak, t_plat_start, t_plat_end, t_end
    loss_non_neg = sum(relu(-non_neg_vars(:))) + alpha;

    loss = loss_time + loss_force + loss_non_neg; % 调整权重平衡
end