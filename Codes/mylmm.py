# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 06:48:09 2020

@author: Alessandro
"""

import matplotlib.pyplot as plt
import numpy as np

# Right hand side of the Cauchy problem
def f(t,y): return -(1+np.tan(t))*y

# Exact solution
def yex(t): return np.exp(-t)*np.cos(t)


def ord_assess(a,b,bm1,nmax):
    c = np.zeros(nmax)
    flag = 1
    ind = np.linspace(0, a.size-1, a.size)
    c[0] = sum(a)
    c[1] = -sum(a*ind)+sum(b) + bm1
    if (c[0] != 1 or c[1] !=1): flag = -1
    for j in range(2, nmax):
        c[j] = sum(a*(-ind)**j)+j*sum(b*(-ind)**(j-1))+j*bm1
        if c[j] == 1: flag = j
    return flag

t0 = 0.0 # initial time
Tf = 1.0 # final time
dt = .02 # time step
p = 2 # number of steps

t_init = np.linspace(t0, (p-1)*dt, p)



y0 = yex(t_init) # initial condition(s)


bm1 = 2./3. # coeff b_{-1}
a = np.array([4./3., -1./3.]) # coefficients
b = np.array([0.0, 0.0]) # coefficients
#bm1 = 1.
#a = np.array([1., 0.])
#b = np.array([0., 0.])

nmax = 5

order = ord_assess(a, b, bm1, nmax)
if order == -1:
    print("The method is not consistent!")
elif order == nmax:
    print("Order estimated ", order, "but test with a larger nmax")
else:
    print("Order of the method ", order)


tol = 1.e-7 # for the nonlinear solver
kmax = 1000 # for the nonlinear solver

i = p
u = np.array(y0)

time = np.array(t_init)


aux_a = np.zeros(p)
aux_b = np.zeros(p)

t_c = time[-1]
epsilon = 1.e-10

ff = f(t_init, y0)

if (bm1 == 0.):
    while (Tf-t_c >= epsilon):
        t_c += dt
        print("Time ", t_c)
        for j in range(0, p):
            aux_a[j] = u[i-j-1]
            aux_b[j] = ff[i-j-1]
        u = np.append(u, sum(a*aux_a) + dt*sum(b*aux_b))
        ff = np.append(ff, f(t_c, u[i]))
        time = np.append(time, t_c)
        i = i + 1

else:
    while (Tf-t_c >= epsilon):
        t_c += dt
        print("Time ", t_c)
        err = 10.
        u_c = u[i-1]
        k = 0
        for j in range(0, p):
            aux_a[j] = u[i-j-1]
            aux_b[j] = ff[i-j-1]

        aux = sum(a*aux_a) + dt*sum(b*aux_b)
        while ((err > tol) & (k < kmax)):
            u_cold = u_c
            u_c = dt*bm1*f(t_c, u_c) + aux
            err = np.abs(u_c - u_cold)
            k = k + 1

        print("Convergence of the nonlinear equation in ",
              k, "iterations, with ", err, "error")
        u = np.append(u, u_c)
        ff = np.append(ff, f(t_c, u_c))
        time = np.append(time, t_c)
        i = i + 1

plt.figure(figsize=(6.5, 4))

plt.plot(time, u)
plt.plot(time, yex(time), 'b+')

print(np.max(np.abs(u-yex(time))))
print(dt)

plt.figure(figsize=(6.5, 4))
plt.plot(time, u-yex(time))

