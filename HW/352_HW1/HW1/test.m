n = 4;
e = ones(n,1);
A = spdiags([e -2*e e],-1:1,n,n);
full(A)

e