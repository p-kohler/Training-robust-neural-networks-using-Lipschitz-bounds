function configure_solver_paths()
% Use configured toolboxes, or retain paths supplied by MATLAB startup.m.
names = {'YALMIP_PATH', 'MOSEK_PATH'};
for k = 1:numel(names)
    folder = getenv(names{k});
    if ~isempty(folder)
        if ~isfolder(folder)
            error('Lipschitz:InvalidToolboxPath', ...
                '%s is not a directory: %s', names{k}, folder);
        end
        addpath(genpath(folder));
    end
end
if exist('sdpvar', 'file') == 0 || exist('optimize', 'file') == 0
    error('Lipschitz:MissingYALMIP', ...
        'YALMIP is unavailable. Set YALMIP_PATH or configure the MATLAB path.');
end
end
