from __future__ import print_function
#from mshr import *
# import matplotlib.pyplot as plt
import numpy as np
from fenics import *

# Create mesh and define function space
mesh = Mesh("./insulating10mm.xml")
boundaries = MeshFunction("size_t", mesh, "./insulating10mm_facet_region.xml")

hmax = mesh.hmax()
hmin = mesh.hmin()

print("Max diameter", hmax) 
print("Min diameter", hmin)

V = FunctionSpace(mesh, 'P', 1)


# Physical Parameters
kappa = Constant(0.05)
alpha = Constant(10)
u_e = Constant(15)
u_w = 60.0

# Define boundary conditions
u_D = Constant(u_w)
chi = alpha
s = alpha*u_e

boundary_conditions = {1: {'Robin': (chi,s)},   # label = 1
                       2: {'Neumann':    0.0}, # label = 2
                       3: {'Neumann':    0.0}, # label = 3
                       4: {'Dirichlet':   u_D}} # label = 4

# Redefine boundary integration measure
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Collect Dirichlet conditions
bcs = DirichletBC(V,u_D,boundaries,4)

#Vector Generic Form (when we have many Dirichlet conditions)
#for i in boundary_conditions:
#    if 'Dirichlet' in boundary_conditions[i]:
#        bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'],
#                         boundaries, i)
#        bcs.append(bc)

 
# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Collect Neumann integrals: in this case, we do not need this, as all the Neumann conditions are homogeneous
# integrals_N = []
# for i in boundary_conditions:
#     if 'Neumann' in boundary_conditions[i]:
#         if boundary_conditions[i]['Neumann'] != 0:
#            g = boundary_conditions[i]['Neumann']
#            integrals_N.append(g*v*ds(i))

# Collect Robin integrals
integrals_R_a = []
integrals_R_L = []
for i in boundary_conditions:
    if 'Robin' in boundary_conditions[i]:
        chi, s = boundary_conditions[i]['Robin']
        integrals_R_a.append(chi*u*v*ds(i))
        integrals_R_L.append(s*v*ds(i))


# Sum integrals to define variational problem
a = kappa*dot(grad(u), grad(v))*dx + sum(integrals_R_a)
L = sum(integrals_R_L) # - sum(integrals_N) + f*v*dx   


# Compute solution
u = Function(V)

# Save solution to file in VTK format
vtkfile = File('insulating5mm.pvd')

fn = FacetNormal(mesh)

solve(a == L, u, bcs, solver_parameters={"linear_solver": "gmres", "preconditioner": "ilu"})

# Save to file and plot solution
vtkfile << (u)
#plot(mesh)
#plot(u)

# post-processing
flux = kappa*dot(grad(u),fn)*ds(1)
total_flux_ins = np.abs(assemble(flux))

# end time loop
print("Total Flux: ",total_flux_ins)
