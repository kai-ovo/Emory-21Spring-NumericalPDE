function [y] = exactSolution(A, y0)

%Convert sparse matrix to full storage
A = full(A);

%Diagonalize A
[T, D] = eig(A);


%Find z_0
z0 = T\y0;

%Build the exact solution y
y = @(t) T*diag(exp(diag(D)*t))*z0;