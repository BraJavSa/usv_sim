function n_angle = normalizeAngle(angle)
    if angle > pi
        angle = angle - 2*pi;
    elseif angle < -pi
        angle = angle + 2*pi;
    else
        % No action needed
    end
    n_angle = angle;
end
