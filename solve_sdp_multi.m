function parameters=solve_sdp_multi(parameters, x, sdp)
%[parameters] = solve_sdp(n_x, n_h, n_y, lambd, mu, Lip, ind_Lip, parameters)
%test

addpath(genpath('..\YALMIP-master'))
addpath(genpath('C:\Program Files\Mosek\9.2'))

%net_dim = x;
% net_dim = [2, 10, 10, 3];

fname = 'weights.json';
weights = jsondecode(fileread(fname));
fn = fieldnames(weights);
for k=1:numel(fn)
    W{k}= weights.(fn{k});
    net_dim(k)=size(W{k},2); % PP: get dimensions from weights
end
net_dim(k+1)=size(W{k},1); % PP: see above
fname = 'biases.json';
biases = jsondecode(fileread(fname));
fn = fieldnames(biases);
for k=1:numel(fn)
    b{k} = biases.(fn{k});
end


l = length(net_dim)-1;
eps = 10^(-9);

lambd = double(sdp.rho);
mu = double(sdp.mu);
ind_Lip = int64(sdp.ind_Lip); % 1 Lipschitz regularizaion, 2 Enforcing Lipschitz bounds
Lip = double(sdp.L_des);
T = squeeze(sdp.T);
t = diag(T);
%for k = 2:l
%    Ts{k} = diag(t(sum(net_dim(2:k-1))+1:sum(net_dim(2:k))));
%end


for k=1:l
    W_bar{k} = sdpvar(net_dim(k+1), net_dim(k));
    Y{k} = parameters.(['Y' num2str(k-1)]);
end

A = [blkdiag(W_bar{1:l-1}), zeros(sum(net_dim(2:l)), sum(net_dim(l:l+1)))];
B = [zeros(sum(net_dim(2:l)), net_dim(1)), eye(sum(net_dim(2:l))), zeros(sum(net_dim(2:l)), net_dim(l+1))];
C = [zeros(net_dim(length(net_dim)), sum(net_dim(1:l-1))), W{l}];

W_diff = 0;
sumOfTraces = 0;
for k=1:l
    W_diff = W_diff + norm(W{k}-W_bar{k}, 'fro')^2 * (lambd/2); % PP: added ^2
    sumOfTraces = sumOfTraces + trace(Y{k}'*(W{k}-W_bar{k})); % PP: added trace()
end


if ind_Lip==1
    rho = sdpvar(1);
    Q_untenrechts = [zeros(net_dim(l)), W_bar{l}'; W_bar{l}, -eye(net_dim(l+1))]; % PP: changed W to W_bar
    Q = blkdiag(-rho*eye(net_dim(1)),zeros(sum(net_dim(2:l-1))), Q_untenrechts);
    %X = blkdiag(zeros(net_dim(1)), Ts{2:l}, zeros(net_dim(l+1)));
    %for k = 2:l
    %    X(sum(net_dim(1:k-2))+1:sum(net_dim(1:k-1)), sum(net_dim(1:k-1))+1:sum(net_dim(1:k))) = W_bar{k-1}'*Ts{k}; % PP: changed W to W_bar
    %    X(sum(net_dim(1:k-1))+1:sum(net_dim(1:k)), sum(net_dim(1:k-2))+1:sum(net_dim(1:k-1))) = Ts{k}*W_bar{k-1}; % PP: changed W to W_bar
    %end
    X2=[A;B]'*[0*T, T; T, -2*T]*[A;B]; % PP: Reformulation of X, check if yours does the same, seemed like factor -2 was missing, adapt in ind_Lip=2

    M = X2 + Q;
    F1 = (M <= -eps*eye(sum(net_dim(1:l+1))));
    optimize(F1, mu*rho + W_diff + sumOfTraces);
elseif ind_Lip==2
    rho = Lip^2;
    Q_untenrechts = [zeros(net_dim(l)), W_bar{l}'; W_bar{l}, -eye(net_dim(l+1))];
    Q = blkdiag(-rho*eye(net_dim(1)),zeros(sum(net_dim(2:l-1))), Q_untenrechts);
%    X = blkdiag(zeros(net_dim(1)), Ts{2:l}, zeros(net_dim(l+1)));
%    for k = 2:l
%        X(sum(net_dim(1:k-2))+1:sum(net_dim(1:k-1)), sum(net_dim(1:k-1))+1:sum(net_dim(1:k))) = W_bar{k-1}'*Ts{k};
%        X(sum(net_dim(1:k-1))+1:sum(net_dim(1:k)), sum(net_dim(1:k-2))+1:sum(net_dim(1:k-1))) = Ts{k}*W_bar{k-1};
%    end
    X2 = [A;B]'*[0*T,T;T-2*T]*[A;B]; %M = X + Q;
    M = X2 + Q;
    F1 = (M <= -eps*eye(sum(net_dim(1:l+1))));
    optimize(F1, mu*rho + W_diff + sumOfTraces);
end

Lipschitz = sqrt(value(rho));

for k=1:l
    W_bar{k} = value(W_bar{k});
    Y{k} = Y{k} + lambd * (W{k}-W_bar{k}); % dual update step
    parameters.(['W' num2str(k-1) '_bar']) = value(W_bar{k}); % PP: changes
    parameters.(['Y' num2str(k-1)]) = Y{k};
end

parameters.Lipschitz=Lipschitz;

