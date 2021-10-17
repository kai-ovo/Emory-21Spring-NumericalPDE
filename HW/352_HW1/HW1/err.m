function [e] = err(U, y)
%
%   U: matrix containing the numerical solution
%   y: the exact solution as a function
%
%   e: the error

%Create the matrix
U = zeros(length(A), n+1);
%Build and identity matrix
I = eye(length(A));

%The two initial values
U(:,1) = y0;
U(:,2) = y1;


%Main loop to calculate the numerical solutions
for j = 2:n
    U(:,j+1) = (I - (dt*5/12)*A)\((I + (Dt*2/3)*A)*U(:,j) - (Dt/12)*A*U(:,j-1));
end