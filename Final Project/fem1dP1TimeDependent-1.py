"""

@author: Alessandro
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy import integrate

def mapd(xh,xi,xip1):
    x = xi + (xip1-xi)*xh
    j = xip1-xi
    return x,j 

def mapi(x,xi,xip1):
    xh = (x-xi)/(xip1-xi)
    ji = 1/(xip1-xi)
    return xh, ji

"""
P1
"""

def phi01(xh):
    # reference function 0 in hat space
    return 1. - xh

def dphi01(xh):
    # reference function 0 in hat space
    return -1. + 0.*xh

def phi11(xh):
    # reference function 1 in hat space
    return xh

def dphi11(xh):
    # reference function 1 in hat space
    return 1.+ 0.*xh



"""
Functions
"""
def f(x, t):
    return  0.*x + 0.

def mu(x, t):
    sigma = 0.1
    return 0.5 * sigma**2 * x**2

def beta(x, t):
    sigma = 0.1
    r = 0.05
    return (sigma**2-r)*x

def sigma(x, t):
    return 1. + 0.*x

def rho(x):
    return 0.05


def bc(a,b,t):
    E = 30
    r = 0.05
    ul = E*np.exp(-r*t)
    ur = 0.0
    return ul, ur


def ic(x):
# initial conditions
#   return np.where(x<=0.3,0.,1.)-np.where(x>=0.7,1.,0.)

    return x


def exact(x,t):
#    return (1-x)*x
    u_ex = x*np.exp(-t)
    du_ex = np.exp(-t) + 0*x
    return u_ex, du_ex


"""
Local Assembly
"""

def loc_asmb1(element,mesh,wg,xhg,t):
    A_l =[[0.,0.],
          [0.,0.]]
    b_l =[[0.],
          [0.]]
    a = mesh[element]
    b = mesh[element+1]
    x,j = mapd(xhg,a,b)
    ji = 1/j
#    print(a,b,x,j,ji)
    A_l[0][0] = sum(wg*(mu(x,t)*dphi01(xhg)*dphi01(xhg)*ji**2 + \
                        beta(x,t)*dphi01(xhg)*phi01(xhg)*ji   + \
                        sigma(x,t)*phi01(xhg)*phi01(xhg))*j)
    A_l[0][1] = sum(wg*(mu(x,t)*dphi11(xhg)*dphi01(xhg)*ji**2 + \
                        beta(x,t)*dphi11(xhg)*phi01(xhg)*ji   + \
                        sigma(x,t)*phi11(xhg)*phi01(xhg))*j)
    A_l[1][0] = sum(wg*(mu(x,t)*dphi01(xhg)*dphi11(xhg)*ji**2 +\
                        beta(x,t)*dphi01(xhg)*phi11(xhg)*ji   + \
                        sigma(x,t)*phi01(xhg)*phi11(xhg))*j)
    A_l[1][1] = sum(wg*(mu(x,t)*dphi11(xhg)*dphi11(xhg)*ji**2 +\
                        beta(x,t)*dphi11(xhg)*phi11(xhg)*ji   + \
                        sigma(x,t)*phi11(xhg)*phi11(xhg))*j)
    b_l[0] =  sum(wg*(f(x,t)*phi01(xhg))*j)
    b_l[1] =  sum(wg*(f(x,t)*phi11(xhg))*j)
#    print(A_l)
    return A_l, b_l    




def mass_loc_asmb1(element,mesh,wg,xhg):
    M_l =[[0.,0.],
          [0.,0.]]
    a = mesh[element]
    b = mesh[element+1]
    x,j = mapd(xhg,a,b)
    M_l[0][0] = sum(wg*(rho(x)*phi01(xhg)*phi01(xhg))*j)
    M_l[0][1] = sum(wg*(rho(x)*phi11(xhg)*phi01(xhg))*j)
    M_l[1][0] = sum(wg*(rho(x)*phi01(xhg)*phi11(xhg))*j)
    M_l[1][1] = sum(wg*(rho(x)*phi11(xhg)*phi11(xhg))*j)
    return M_l    


def mu_upw(x,h):
    return mu(x)+np.abs(beta(x))*h/2

def mu_sg(x,h):
    Pe = np.abs(beta(x))*h/(2*mu(x))
    return mu(x)*(Pe+2*Pe/(np.exp(2*Pe)-1))



def loc_asmb1UPW(element,mesh,wg,xhg,t):
    A_l =[[0.,0.],
          [0.,0.]]
    b_l =[[0.],
          [0.]]
    a = mesh[element]
    b = mesh[element+1]
    h = b-a    
    x,j = mapd(xhg,a,b)
    ji = 1/j
#    print(a,b,x,j,ji)
    A_l[0][0] = sum(wg*(mu_upw(x,h)*dphi01(xhg)*dphi01(xhg)*ji**2 + \
                        beta(x)*dphi01(xhg)*phi01(xhg)*ji   + \
                        sigma(x)*phi01(xhg)*phi01(xhg))*j)
    A_l[0][1] = sum(wg*(mu_upw(x,h)*dphi11(xhg)*dphi01(xhg)*ji**2 + \
                        beta(x)*dphi11(xhg)*phi01(xhg)*ji   + \
                        sigma(x)*phi11(xhg)*phi01(xhg))*j)
    A_l[1][0] = sum(wg*(mu_upw(x,h)*dphi01(xhg)*dphi11(xhg)*ji**2 +\
                        beta(x)*dphi01(xhg)*phi11(xhg)*ji   + \
                        sigma(x)*phi01(xhg)*phi11(xhg))*j)
    A_l[1][1] = sum(wg*(mu_upw(x,h)*dphi11(xhg)*dphi11(xhg)*ji**2 +\
                        beta(x)*dphi11(xhg)*phi11(xhg)*ji   + \
                        sigma(x)*phi11(xhg)*phi11(xhg))*j)
    b_l[0] =  sum(wg*(f(x,t)*phi01(xhg))*j)
    b_l[1] =  sum(wg*(f(x,t)*phi11(xhg))*j)
#    print(A_l)
    return A_l, b_l    



def loc_asmb1SG(element,mesh,wg,xhg,t):
    A_l =[[0.,0.],
          [0.,0.]]
    b_l =[[0.],
          [0.]]
    a = mesh[element]
    b = mesh[element+1]
    h = b-a    
    x,j = mapd(xhg,a,b)
    ji = 1/j
#    print(a,b,x,j,ji)
    A_l[0][0] = sum(wg*(mu_sg(x,h)*dphi01(xhg)*dphi01(xhg)*ji**2 + \
                        beta(x)*dphi01(xhg)*phi01(xhg)*ji   + \
                        sigma(x)*phi01(xhg)*phi01(xhg))*j)
    A_l[0][1] = sum(wg*(mu_sg(x,h)*dphi11(xhg)*dphi01(xhg)*ji**2 + \
                        beta(x)*dphi11(xhg)*phi01(xhg)*ji   + \
                        sigma(x)*phi11(xhg)*phi01(xhg))*j)
    A_l[1][0] = sum(wg*(mu_sg(x,h)*dphi01(xhg)*dphi11(xhg)*ji**2 +\
                        beta(x)*dphi01(xhg)*phi11(xhg)*ji   + \
                        sigma(x)*phi01(xhg)*phi11(xhg))*j)
    A_l[1][1] = sum(wg*(mu_sg(x,h)*dphi11(xhg)*dphi11(xhg)*ji**2 +\
                        beta(x)*dphi11(xhg)*phi11(xhg)*ji   + \
                        sigma(x)*phi11(xhg)*phi11(xhg))*j)
    b_l[0] =  sum(wg*(f(x,t)*phi01(xhg))*j)
    b_l[1] =  sum(wg*(f(x,t)*phi11(xhg))*j)
#    print(A_l)
    return A_l, b_l    




"""
Error Computation
"""

def error_h1l2(x,wg,xhg,u,t):
    Np1 = np.size(x)
    errl2 = 0.0
    errh1 = 0.0
    for el in range(0,Np1-1):
        xg, j = mapd(xhg, x[el], x[el+1])
        ji = 1/j
        ue,due = exact(xg,t)
        errl2 += sum(wg*j*(u[el]*phi01(xhg)+u[el+1]*phi11(xhg)-ue)**2)
        errh1 += sum(wg*j*(u[el]*dphi01(xhg)*ji+u[el+1]*dphi11(xhg)*ji-due)**2)
#        xx = np.linspace(x[el],x[el+1],q+1)
#        xh,ji = mapi(xx,x[el],x[el+1])
#        uc = u[el]*dphi01(xh) + u[el+1]*dphi11(xh)
#        plt.plot(xx,uc, 'b')
#    plt.show()   
    errh1 += errl2
    errl2 = np.sqrt(errl2)
    errh1 = np.sqrt(errh1)
    return errl2, errh1


def plot_fine_u(x,q,wg,xhg,u,t):
    Np1 = np.size(x)
    for el in range(0,Np1-1):
        xx = np.linspace(x[el],x[el+1],q+1)
        xh,ji = mapi(xx,x[el],x[el+1])
        uc = u[el]*phi01(xh) + u[el+1]*phi11(xh)
        uex,duex = exact(xx,t)
        plt.plot(xx,uc, 'b')
        plt.plot(xx,uex,'r') 

        
    plt.show()   
    return 0


def plot_fine_du(x,q,wg,xhg,u,t):
    Np1 = np.size(x)
    for el in range(0,Np1-1):
        xx = np.linspace(x[el],x[el+1],q+1)
        xh,ji = mapi(xx,x[el],x[el+1])
        uc = u[el]*dphi01(xh)*ji + u[el+1]*dphi11(xh)*ji
        uex,duex = exact(xx,t)
        plt.plot(xx,uc, 'b')
        plt.plot(xx,duex,'r') 

    plt.show()   
    return 0

def plot_error_u(x,q,wg,xhg,u,t):
    Np1 = np.size(x)
    for el in range(0,Np1-2,2):
        xx = np.linspace(x[el],x[el+2],q+1)
        xh,ji = mapi(xx,x[el],x[el+2])
        uc = u[el]*phi01(xh) + u[el+1]*phi11(xh) 
        uex,duex = exact(xx,t)
        plt.plot(xx,uc-uex, 'k')

        
    plt.show()   
    return 0

def plot_error_du(x,q,wg,xhg,u,t):
    Np1 = np.size(x)
    for el in range(0,Np1-2,2):
        xx = np.linspace(x[el],x[el+2],q+1)
        xh,ji = mapi(xx,x[el],x[el+2])
        uc = u[el]*dphi01(xh)*ji + u[el+1]*dphi11(xh)*ji 
        uex,duex = exact(xx,t)
        plt.plot(xx,uc-duex, 'k')

    plt.show()   
    return 0






def visualize(x, t, u):
    plt.plot(x, u, 'r')
#    plt.plot(x,u_ex(x, t),'b')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution at time t=%g' % t)
#    umin = 1.2*u.min()
#    umax = -umin
    plt.axis([x[0], x[-1], -0.5, 1.2])
    plt.show()


"""
MAIN
"""
#plt.plot(x,phi01(x))
#plt.plot(x,phi11(x))
#plt.show()
#plt.plot(x,phi02(x))
#plt.plot(x,phi12(x))
#plt.plot(x,phi22(x))
#plt.show()
#plt.plot(x,phi03(x))
#lt.plot(x,phi13(x))
#plt.plot(x,phi23(x))
#plt.plot(x,phi33(x))
#plt.show()

# INTEGRATION
# Gauss-Legendre (default interval is [-1, 1])
deg = 10
xr, w = np.polynomial.legendre.leggauss(deg+1)
# Translate x values from the interval [-1, 1] to [0, 1]
xhg = 0.5*(xr + 1)
wg = w/2


mass_lumping = 0

### MASS LUMPING
if (mass_lumping):
    print('Mass Lumping ON')
    xhg = np.zeros(2)
    xhg[0] = 0.
    xhg[1] = 1.
    wg = np.zeros(2)
    wg[0] = 0.5
    wg[1] = 0.5
    print(xhg)
    print(wg)


#print(xhg,w)
#gauss = sum(wg * np.sin(xhg))
#print(gauss)
#print(-np.cos(1)+1)

# PROBLEM DEFINITION

a = 0
b = 100
h = 1
theta = 0.5
t0 = 0.0
Tf = 1.
dt = 0.01
dti = 1./dt
aux = 1.

nsteps = int(round((Tf-t0)/dt))
print(nsteps)
# Error computing
q = 100 # quotient fine/coarse
errH1 = np.zeros(nsteps)
errL2 = np.zeros(nsteps)


t=t0

N = int((b-a)/h)
#print(N)

# MESHING
x = np.linspace(a,b,N+1)
#print(x)    
rhs = np.zeros(N+1)

A =sp.diags([0., 0., 0.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')

M =sp.diags([0., 0., 0.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')


LeftMatrix =sp.diags([0., 0., 0.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')
RightMatrix =sp.diags([0., 0., 0.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')



u   = np.zeros(N+1)
u_n = np.zeros(N+1)


# MASS ASSEMBLY
for el in range(0,N):
    M_l = mass_loc_asmb1(el,x,wg,xhg)
    M[el,el] += M_l[0][0]
    M[el,el+1] += M_l[0][1]
    M[el+1,el] += M_l[1][0]
    M[el+1,el+1] += M_l[1][1]
#    A_l, b_l = loc_asmb1(el,x,wg,xhg,t0)
#    A[el,el] += A_l[0][0]
#    A[el,el+1] += A_l[0][1]
#    A[el+1,el] += A_l[1][0]
#    A[el+1,el+1] += A_l[1][1]    
#    rhs[el] += b_l[0]
#    rhs[el+1] += b_l[1]


rhsold = rhs

u_n = ic(x)
#visualize(x,0,u_n)
plot_fine_u(x,q,wg,xhg,u_n,t0)
#    plot_error_u(x,q,wg,xhg,u,t)
plot_fine_du(x,q,wg,xhg,u_n,t0)


# TIME LOOP

for k in range(0,nsteps):
    t+=dt
    print(t)
    # ASSEMBLY
    for el in range(0,N):
        A_l, b_l = loc_asmb1(el,x,wg,xhg,t)
        A[el,el] += A_l[0][0]
        A[el,el+1] += A_l[0][1]
        A[el+1,el] += A_l[1][0]
        A[el+1,el+1] += A_l[1][1]
        rhs[el] += b_l[0]
        rhs[el+1] += b_l[1]

    rhsaux = rhs
    rhs = dt*theta*rhs + dt*(1-theta)*rhsold
    rhsold = rhsaux
    LeftMatrix = M + dt*theta*A
    RightMatrix = M - dt*(1-theta)*A
    rhs += RightMatrix*u_n 
# boundary conditions
    uL, uR = bc(a, b, t)
    LeftMatrix[0, 1] = 0.
    LeftMatrix[-1, -2] = 0.
    LeftMatrix[0, 0] = aux
    LeftMatrix[-1, -1] = aux

    rhs[0] = uL*aux
    rhs[-1] = uR*aux
#    plt.spy(LeftMatrix)
#    plt.show()
# SOLVING
# Linear system solving
#   print(rhs)
#    print(LeftMatrix)
    u = sp.linalg.spsolve(LeftMatrix, rhs)
#    visualize(x,t,u)
    u_n = u
    errL2[k],errH1[k] = error_h1l2(x,wg,xhg,u,t)
    print(errL2[k],errH1[k])
#CLEANING    
    rhs = np.zeros(N+1)
    A =sp.diags([0., 0., 0.], [-1, 0, 1], shape=[N+1, N+1], format = 'csr')
    LeftMatrix = A
    RightMatrix = A
# POSTPROCESSING
    plot_fine_u(x,q,wg,xhg,u,t)
    plot_error_u(x,q,wg,xhg,u,t)
    plot_fine_du(x,q,wg,xhg,u,t)
    plot_error_du(x,q,wg,xhg,u,t)


print('errLinf(L2)',np.max(errL2))
print('errL2(H1)',0.5*dt*(errH1[0]+errH1[-1])+dt*sum(errH1[1:-2])) #Composite trapezoidal rule
    

