"""
MATH572 Parabolic Problems

@author: Alessandro
"""
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg

"""
===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
T     The stop time for the simulation.
I     Initial condition (Python function of x).
a     Variable coefficient (constant).
L     Length of the domain ([0,L]).
x     Mesh points in space.
t     Mesh points in time.
n     Index counter in time.
u     Unknown at current/new time level.
u_n   u at the previous time level.
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================
Equation: (already rewritten asn an IVP)
u_t - 0.5*sigma^2*x^2*u_xx - r*x*u_x + ru = 0 
Call Option
I.C. u(x,0) = max(0,x-E)
B.C. u(0,t)=0, u(x_M,t) = x_M - E*exp(-r*t)

"""

def visualize(x, t, u):
    plt.plot(x, u, 'r')
#    plt.plot(x,u_ex(x, t),'b')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution at time t=%g' % t)
#    umin = 1.2*u.min()
#    umax = -umin
    #plt.axis([x[0], x[-1], -0.1, 150.])
    plt.show()



def IC(x,E):
# initial conditions
    N = x.size
    v = np.zeros(N)
    for i in range(0,N):
        v[i] = max(x[i]-E,0.) # call
        #v[i] = max(E-x[i],0.) # put
        
    return v
#    return np.sin(np.pi*x)

def BC(a,b,t,E,r):
    uL = 0. # call
    uR = b - E*np.exp(-r*t) # call
    #uL = E*np.exp(-r*t)  # put
    #uR = 0. # put
    return uL, uR

def f(x,t):
    return 0.0


# DEFINITION OF THE PROBLEM
T = 1. # final time
xl = 0. # leftmost point
xr = 100. # righmost point (xM)

sigma = 0.1 # volatility
r = 0.05
E = 30.


# discretization parameters
dt = 0.01
h = 1.
th = 1. # theta

Nx = int(round(np.abs(xr-xl)/h))
Nt = int(round(T/dt))

# Space mesh & time mesh
x = np.linspace(xl,xr,Nx+1)
t = np.linspace(0,T,Nt+1)

# Error computing data structures
eil = np.zeros(Nt)
el2l = np.zeros(Nt)


# MATRIX ASSEMBLY (for time independent coefficients)
u   = np.zeros(Nx+1)
u_n = np.zeros(Nx+1)


# Data structures for the linear system
A = np.zeros((Nx+1, Nx+1))
b = np.zeros(Nx+1)

Ad =sp.diags([1., -2., 1.], [-1, 0, 1], shape=[Nx+1, Nx+1], format = 'csr') #basic discretization
Ac =sp.diags([-1., 0., 1.], [-1, 0, 1], shape=[Nx+1, Nx+1], format = 'csr') #basic discretization
Ar = r*sp.identity(Nx+1, format = 'csr')

print(Ad)

for i in range(1,Nx):
    for j in range(i-1,i+2):
        print(i,j)
        Ad[i,j] = -0.5*(sigma**2)*(x[i]**2)/h**2*Ad[i,j]
        Ac[i,j] = -r*x[i]/(2*h)*Ac[i,j]

Ad[0, 0] = -0.5*(sigma**2)*x[0]**2/h**2*Ad[0,0]
Ad[0, 1] = -0.5*(sigma**2)*x[0]**2/h**2*Ad[0,1]
Ad[-1,-2] = -0.5*(sigma**2)*x[-1]**2/h**2*Ad[-1,-2]
Ad[-1,-1] = -0.5*(sigma**2)*x[-1]**2/h**2*Ad[-1,-1]

print(Ad)

Ac[0, 0] =  -r*x[0]/(2*h)*Ac[0,0]
Ac[0, 1] =  -r*x[0]/(2*h)*Ac[0,1]
Ac[-1,-2] = -r*x[-1]/(2*h)*Ac[-1,-2]
Ac[-1,-1] = -r*x[-1]/(2*h)*Ac[-1,-1]


A = Ad + Ac + Ar

LeftM = sp.identity(Nx+1, format = 'csr') + dt*th*A
RightM = sp.identity(Nx+1, format = 'csr') - dt*(1-th)*A

# boundary conditions
aux_bc = 1.
LeftM[0, 1] = 0.
LeftM[-1, -2] = 0.
LeftM[0, 0] = aux_bc
LeftM[-1, -1] = aux_bc


# REM: We could factorize the matrix LeftM here

# Initial conditions
tc = 0 # current time
u_n = IC(x, E)
visualize(x,tc,u_n)

# TIME LOOP
for n in range(0,Nt):
    tc += dt
    print("Computing at time", tc)
    # right hand side
    b = dt*(th*f(x,tc)+(1-th)*f(x,tc-dt)) + RightM*u_n
    uL, uR = BC(xl,xr,tc, E, r)
    b[0] = aux_bc*uL
    b[-1] = aux_bc*uR
    u = sp.linalg.spsolve(LeftM, b)
    print(u)
    visualize(x,tc,u)
    u_n = u

