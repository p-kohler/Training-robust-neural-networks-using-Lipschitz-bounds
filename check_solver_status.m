function check_solver_status(diagnostics, context)
% Never consume values from a failed or indeterminate optimization.
if diagnostics.problem ~= 0
    error('Lipschitz:SolverFailure', '%s failed (YALMIP code %d): %s', ...
        context, diagnostics.problem, diagnostics.info);
end
end
