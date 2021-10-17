function [U,e] = Heun(A, y0, n, Dt, yex)
%
%   A:  the matrix
%   y0: the initial condition
%   n: the # of time steps
%   Dt: time step
%   yex: the exact solution
%
%   U: matrix contaning each of the solutions as columns
%   e: the error associated with this method
%

%Create the matrix
U = zeros(length(A), n+1);
%Build and identity matrix
I = eye(length(A));
%The vector that will stor
err = zeros(n,1);
%A time variable to calculate the exact solution
t = Dt;

%The initial values
U(:,1) = y0;

%Matrix that is used several times in the loop
M = I + Dt*A+ ((Dt^2)/2)*A*A;

%Main loop to calculate the numerical solutions, and find the errors
for i = 1:n
    U(:,i+1) = M*U(:,i);
    err(i) = max(abs(yex(t) - U(:,i+1)));
    t = t + Dt;
end

%return the error
e = max(err);