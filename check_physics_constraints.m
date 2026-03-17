function check_physics_constraints(Y_pred)
    % 约束条件检查
    violations = sum(Y_pred(:,2) >= Y_pred(:,4) | ...    % t_peak < t_plat_start
                    Y_pred(:,4) >= Y_pred(:,5) | ...     % t_plat_start < t_plat_end
                    Y_pred(:,5) >= Y_pred(:,6) | ...     % t_plat_end < t_end
                    Y_pred(:,3) >= Y_pred(:,1), 2);      % F_plat_start < F_peak
    
    violation_rate = mean(violations > 0) * 100;
    fprintf('物理约束违反率: %.2f%%\n', violation_rate);
    
    % 可视化异常样本
    if violation_rate > 0
        figure;
        scatter3(Y_pred(:,1), Y_pred(:,2), Y_pred(:,5), 40, violations, 'filled');
        xlabel('t_{peak}'); ylabel('t_{plat\_start}'); zlabel('F_{plat\_start}');
        title(sprintf('物理约束违反样本 (%.1f%%)', violation_rate));
        colorbar;
    end
end
