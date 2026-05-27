function designLevel1ControllerTuning(uav)
% GAINTUNINGGUI  Interactive GUI for Level-1 PID gain tuning.
%
%   gainTuningGUI()
%
%   Launches a graphical interface for tuning the Level-1 angular-rate PID
%   gains and visualizing the closed-loop eigenvalues of Ac1 across five
%   operating conditions simultaneously.
%
% OPERATING CONDITIONS (one per eigenvalue plot)
%   1. wd = -wdmax,  T = Tmin
%   2. wd = -wdmax,  T = Tmax
%   3. wd =  0,      T = Thover
%   4. wd = +wdmax,  T = Tmin
%   5. wd = +wdmax,  T = Tmax
%
% USAGE
%   - Edit the nine gain boxes (Kp, Ki, Kd for x, y, z axes).
%   - Press "Update Gains" to apply gains and refresh all eigenvalue plots.
%   - Edit plot settings and press "Update View" to adjust axis limits and
%     overlay target damping / decay lines.
%
% DEPENDENCIES
%   SETDEFAULTPARAMS, COMPUTEMINTHRUSTWORSTCASE, LINEARIZEDMATRICESLEVEL1
%
% See also: LINEARIZEDMATRICESLEVEL1, SETDEFAULTPARAMS,
%           COMPUTEMINTHRUSTWORSTCASE, DESIGNLEVEL1CONTROLLERILMI

% -------------------------------------------------------------------------
% Copyright (c) 2026 Ahmet Taha Koru
% SPDX-License-Identifier: MIT
% -------------------------------------------------------------------------

    %% --- Initialization -------------------------------------------------
    uav.Tmin = computeMinThrustWorstCase(uav, 30);

    %% --- Operating condition definitions --------------------------------
    conditions = {
        '-wdmax / Tmin',   -uav.wdmax,  uav.Tmin   ;
        '-wdmax / Tmax',   -uav.wdmax,  uav.Tmax   ;
        ' 0     / Thover',  zeros(3,1), uav.Thover  ;
        '+wdmax / Tmin',    uav.wdmax,  uav.Tmin   ;
        '+wdmax / Tmax',    uav.wdmax,  uav.Tmax   ;
    };

    nCond = size(conditions, 1);

    %% --- Default view settings ------------------------------------------
    viewSettings.xlim        = [-150,  0  ];
    viewSettings.ylim        = [-150,  150 ];
    viewSettings.damping      = 0.707;
    viewSettings.decay        = 0.1;
    viewSettings.showDamping  = true;
    viewSettings.showDecay    = true;

    %% --- Figure layout --------------------------------------------------
    fig = figure('Name',        'Level-1 Gain Tuning', ...
                 'NumberTitle', 'off',                  ...
                 'Position',    [80, 80, 1400, 750],    ...
                 'Resize',      'on');

    % =====================================================================
    % Left panel: gain controls
    % =====================================================================
    ctrlPanel = uipanel(fig,                     ...
        'Title',      'PID Gains',               ...
        'FontSize',   11,                         ...
        'FontWeight', 'bold',                     ...
        'Position',   [0.01, 0.01, 0.16, 0.98]);

    gainNames  = {'kp\_wx', 'kp\_wy', 'kp\_wz', ...
                  'ki\_wx', 'ki\_wy', 'ki\_wz', ...
                  'kd\_wx', 'kd\_wy', 'kd\_wz'};

    gainFields = {'kp_wx', 'kp_wy', 'kp_wz', ...
                  'ki_wx', 'ki_wy', 'ki_wz', ...
                  'kd_wx', 'kd_wy', 'kd_wz'};

    gainDefaults = [uav.kp_wx, uav.kp_wy, uav.kp_wz, ...
                    uav.ki_wx, uav.ki_wy, uav.ki_wz, ...
                    uav.kd_wx, uav.kd_wy, uav.kd_wz];

    nGains   = numel(gainNames);
    rowH     = 0.082;
    topStart = 0.93;
    gainBoxes = gobjects(nGains, 1);

    for k = 1:nGains
        yPos = topStart - (k - 1) * rowH;
        uicontrol(ctrlPanel,                              ...
            'Style',               'text',               ...
            'String',              gainNames{k},          ...
            'Units',               'normalized',          ...
            'Position',            [0.04, yPos, 0.52, 0.06], ...
            'HorizontalAlignment', 'left',               ...
            'FontSize',            10);
        gainBoxes(k) = uicontrol(ctrlPanel,              ...
            'Style',    'edit',                          ...
            'String',   num2str(gainDefaults(k), '%g'),  ...
            'Units',    'normalized',                    ...
            'Position', [0.56, yPos, 0.40, 0.06],       ...
            'FontSize', 10);
    end

    uicontrol(ctrlPanel,                          ...
        'Style',           'pushbutton',          ...
        'String',          'Update Gains',        ...
        'Units',           'normalized',          ...
        'Position',        [0.08, 0.02, 0.84, 0.07], ...
        'FontSize',        11,                    ...
        'FontWeight',      'bold',                ...
        'BackgroundColor', [0.2, 0.6, 0.2],      ...
        'ForegroundColor', 'white',               ...
        'Callback',        @onUpdateGains);

    % =====================================================================
    % View settings panel
    % =====================================================================
    viewPanel = uipanel(fig,                      ...
        'Title',      'Plot Settings',            ...
        'FontSize',   11,                         ...
        'FontWeight', 'bold',                     ...
        'Position',   [0.18, 0.01, 0.14, 0.98]);

    viewLabels   = {'X min', 'X max', 'Y min', 'Y max', ...
                    'Target Damping', 'Target Decay'};
    viewDefaults = [viewSettings.xlim(1), viewSettings.xlim(2), ...
                    viewSettings.ylim(1), viewSettings.ylim(2), ...
                    viewSettings.damping, viewSettings.decay];

    nView    = numel(viewLabels);
    viewBoxes = gobjects(nView, 1);

    for k = 1:nView
        yPos = topStart - (k - 1) * rowH;
        uicontrol(viewPanel,                              ...
            'Style',               'text',               ...
            'String',              viewLabels{k},         ...
            'Units',               'normalized',          ...
            'Position',            [0.04, yPos, 0.56, 0.06], ...
            'HorizontalAlignment', 'left',               ...
            'FontSize',            10);
        viewBoxes(k) = uicontrol(viewPanel,              ...
            'Style',    'edit',                          ...
            'String',   num2str(viewDefaults(k), '%g'),  ...
            'Units',    'normalized',                    ...
            'Position', [0.60, yPos, 0.36, 0.06],       ...
            'FontSize', 10);
    end

    % Checkboxes: enable/disable overlays
    dampingCheck = uicontrol(viewPanel,                   ...
        'Style',   'checkbox',                            ...
        'String',  'Show damping cone',                   ...
        'Value',   viewSettings.showDamping,              ...
        'Units',   'normalized',                          ...
        'Position',[0.04, topStart - nView*rowH, 0.92, 0.06], ...
        'FontSize', 9);

    decayCheck = uicontrol(viewPanel,                     ...
        'Style',   'checkbox',                            ...
        'String',  'Show decay line',                     ...
        'Value',   viewSettings.showDecay,                ...
        'Units',   'normalized',                          ...
        'Position',[0.04, topStart - (nView+1)*rowH, 0.92, 0.06], ...
        'FontSize', 9);

    uicontrol(viewPanel,                          ...
        'Style',           'pushbutton',          ...
        'String',          'Update View',         ...
        'Units',           'normalized',          ...
        'Position',        [0.08, 0.02, 0.84, 0.07], ...
        'FontSize',        11,                    ...
        'FontWeight',      'bold',                ...
        'BackgroundColor', [0.2, 0.4, 0.7],      ...
        'ForegroundColor', 'white',               ...
        'Callback',        @onUpdateView);

    % =====================================================================
    % Eigenvalue plot axes
    % =====================================================================
    plotAxes  = gobjects(nCond, 1);
    nCols     = 3;
    nRows     = 2;
    leftOff   = 0.34;
    plotW     = (1 - leftOff - 0.02) / nCols;
    plotH     = (1 - 0.06) / nRows;

    for k = 1:nCond
        col = mod(k - 1, nCols);
        row = nRows - 1 - floor((k - 1) / nCols);
        plotAxes(k) = axes(fig,                              ...
            'Units',    'normalized',                        ...
            'Position', [leftOff + col*plotW + 0.01,        ...
                         0.06  + row*plotH + 0.01,          ...
                         plotW - 0.02, plotH - 0.06]);  %#ok<LAXES>
        hold(plotAxes(k), 'on');
        grid(plotAxes(k), 'on');
        xlabel(plotAxes(k), 'Real');
        ylabel(plotAxes(k), 'Imag');
    end

    %% --- Initial render -------------------------------------------------
    updatePlots();

    % =====================================================================
    % Callbacks
    % =====================================================================
    function onUpdateGains(~, ~)
        for j = 1:nGains
            val = str2double(gainBoxes(j).String);
            if isnan(val)
                uiwait(msgbox( ...
                    sprintf('Invalid value for %s — please enter a number.', gainFields{j}), ...
                    'Input Error', 'error'));
                return;
            end
    
            % P gains: indices 1-3 (kp_wx, kp_wy, kp_wz)
            % I gains: indices 4-6 (ki_wx, ki_wy, ki_wz)
            if j <= 6 && val <= 0
                uiwait(msgbox( ...
                    sprintf('%s must be strictly positive (got %.4g).', gainFields{j}, val), ...
                    'Input Error', 'error'));
                return;
            end
    
            % D gains: indices 7-9 (kd_wx, kd_wy, kd_wz) — allow zero
            if j >= 7 && val < 0
                uiwait(msgbox( ...
                    sprintf('%s must be non-negative (got %.4g).', gainFields{j}, val), ...
                    'Input Error', 'error'));
                return;
            end
    
            uav.(gainFields{j}) = val;
        end
        uav.Kpw = diag([uav.kp_wx, uav.kp_wy, uav.kp_wz]);
        uav.Kiw = diag([uav.ki_wx, uav.ki_wy, uav.ki_wz]);
        uav.Kdw = diag([uav.kd_wx, uav.kd_wy, uav.kd_wz]);
        updatePlots();
    end

    function onUpdateView(~, ~)
        vals = zeros(nView, 1);
        viewFieldNames = {'xlim1','xlim2','ylim1','ylim2','damping','decay'};
        for j = 1:nView
            vals(j) = str2double(viewBoxes(j).String);
            if isnan(vals(j))
                uiwait(msgbox( ...
                    sprintf('Invalid value for %s.', viewLabels{j}), ...
                    'Input Error', 'error'));
                return;
            end
        end
        viewSettings.xlim       = [vals(1), vals(2)];
        viewSettings.ylim       = [vals(3), vals(4)];
        viewSettings.damping    = vals(5);
        viewSettings.decay      = vals(6);
        viewSettings.showDamping = logical(dampingCheck.Value);
        viewSettings.showDecay   = logical(decayCheck.Value);
        updatePlots();
    end

    % =====================================================================
    % Core plot update
    % =====================================================================
    function updatePlots()
        for k = 1:nCond
            wd = conditions{k, 2};
            T  = conditions{k, 3};
            ax = plotAxes(k);
            cla(ax);
            hold(ax, 'on');

            try
                [~, ~, ~, ~, Ac1] = linearizedMatricesLevel1(uav, wd, T);
                ev = eig(Ac1);

                % Eigenvalue scatter
                plot(ax, real(ev), imag(ev), 'b.', 'MarkerSize', 16);

                % Imaginary axis
                xline(ax, 0, 'k--', 'LineWidth', 0.8);

                % Target decay line
                if viewSettings.showDecay
                    xline(ax, -viewSettings.decay, ...
                        'Color', [0.8, 0.4, 0.0],  ...
                        'LineStyle', '--',           ...
                        'LineWidth', 1.2,            ...
                        'Label', sprintf('\\sigma = %.3f', viewSettings.decay), ...
                        'LabelVerticalAlignment', 'bottom');
                end

                % Target damping cone
                if viewSettings.showDamping
                    zeta  = viewSettings.damping;
                    theta = acos(zeta);          % cone half-angle
                    yLim  = viewSettings.ylim;
                    xLim  = viewSettings.xlim;
                    rMax  = max(abs([xLim, yLim])) * 2;
                    % Upper and lower cone boundary lines through origin
                    slope = tan(pi/2 - theta);   % Im/Re ratio at angle theta
                    xPts  = [0, -rMax];
                    plot(ax, xPts,  slope * (-xPts), ...
                        'Color', [0.7, 0.1, 0.7],    ...
                        'LineStyle', '--', 'LineWidth', 1.2);
                    plot(ax, xPts, -slope * (-xPts), ...
                        'Color', [0.7, 0.1, 0.7],    ...
                        'LineStyle', '--', 'LineWidth', 1.2);
                    % Fill the admissible cone region
                    xFill = [0, -rMax, -rMax, 0];
                    yFill = [0,  slope*rMax, -slope*rMax, 0];
                    fill(ax, xFill, yFill, [0.7, 0.1, 0.7], ...
                        'FaceAlpha', 0.06, 'EdgeColor', 'none');
                    text(ax, mean(xLim)*0.6, 0, ...
                        sprintf('\\zeta = %.3f', zeta), ...
                        'Color', [0.7, 0.1, 0.7], 'FontSize', 8, ...
                        'HorizontalAlignment', 'center');
                end

                % Stability label
                if max(real(ev)) < 0
                    labelColor = [0.1, 0.55, 0.1];
                    labelStr   = 'STABLE';
                else
                    labelColor = [0.8, 0.1, 0.1];
                    labelStr   = 'UNSTABLE';
                end
                title(ax, sprintf('%s  [%s]', conditions{k,1}, labelStr), ...
                    'Color', labelColor, 'FontSize', 9);

            catch ME
                title(ax, sprintf('%s  [ERROR]', conditions{k,1}), ...
                    'Color', [0.8, 0.1, 0.1], 'FontSize', 9);
                warning('gainTuningGUI:plotError', ...
                    'Condition %d: %s', k, ME.message);
            end

            xlim(ax, viewSettings.xlim);
            ylim(ax, viewSettings.ylim);
            xlabel(ax, 'Real');
            ylabel(ax, 'Imag');
            grid(ax, 'on');
            drawnow;
        end
    end

end
