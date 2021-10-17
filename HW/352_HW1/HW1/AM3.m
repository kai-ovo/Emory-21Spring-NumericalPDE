function [U,e] = AM3(A, y0, y1, n, Dt, yex)
%
%   A:  the matrix
%   y0, y1: the 2 initial conditions
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
err = zeros(n-1,1);
%A time variable to calculate the exact solution
t = 2*Dt;

%The two initial values
U(:,1) = y0;
U(:,2) = y1;

%Matrices that are used several times in the loop
M1 = I - (Dt*5/12)*A;
M2 = (I + (Dt*2/3)*A);
M3 = (Dt/12)*A;


%Main loop to calculate the numerical solutions, and find the errors
for i = 2:n
    U(:,i+1) = M1\(M2*U(:,i) - M3*U(:,i-1));
    err(i-1) = max(abs(yex(t) - U(:,i+1)));
    t = t + Dt;
end

%return the error
e = max(err);