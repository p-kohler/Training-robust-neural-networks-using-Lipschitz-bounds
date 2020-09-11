% Dieser Funktion soll net_dim als Array, alle weight-Matrizen als
% cell_Array, rho und T übergeben werden.

function Lip = calculate_Lipschitz_multi(parameters, x)

addpath(genpath('..\YALMIP-master'))
addpath(genpath('C:\Program Files\Mosek\9.0'))

net_dim = x;
% net_dim = [2, 10, 10, 3];
l = length(net_dim)-1;
eps = 10^(-9);

fname = 'weights.json';
weights = jsondecode(fileread(fname));
fn = fieldnames(weights);
for k=1:numel(fn)
    W{k}= weights.(fn{k});
end
fname = 'biases.json';
biases = jsondecode(fileread(fname));
fn = fieldnames(biases);
for k=1:numel(fn)
    b{k} = biases.(fn{k});
end

T = diag(sdpvar(sum(net_dim(2:l)),1));
rho = sdpvar(1); % rho=L^2

A = [blkdiag(W{1:l-1}), zeros(sum(net_dim(2:l)), net_dim(l))];
B = [zeros(sum(net_dim(2:l)), net_dim(1)), eye(sum(net_dim(2:l)))];
% C = [zeros(net_dim(length(net_dim)), sum(net_dim(1:l-1))), W{l}];

Q = blkdiag(-rho*eye(net_dim(1)),zeros(sum(net_dim(2:l-1))),W{l}'*W{l});
M = [A; B]' * [zeros(sum(net_dim(2:l))), T; T, -2*T] * [A; B] + Q;
F1 = (M <= -eps*eye(sum(net_dim(1:l))));

optimize(F1, rho);
Lip.Lipschitz = sqrt(double(rho));
Lip.T = double(T);

L=Lip.Lipschitz;
T=Lip.T;