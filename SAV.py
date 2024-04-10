from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def f(u):
    return (u-u*u*u)
def F(u):
    return 0.25*(1-u*u)*(1-u*u)

# define the parameters
epsilon = 0.01
T  = 20
dt = 0.1
t = 0.0
C0 = 1.0


# initialize variables
gamma = 100.0
r = 100.0
r0= 100.0
r1= 100.0
S = []
t_axis = []

# create files for storing the results
file = XDMFFile('./results/SAV.xdmf')

# create mesh
x1 = Point(-1, -1)
x2 = Point(1, 1)
mesh = RectangleMesh(x1, x2, 128, 128)

# define the boundary
# class boundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary        
# bd = boundary()
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

    def map(self, x, y):
        if near(x[0], 1.0):
            y[0] = x[0] - 2.0
            y[1] = x[1]
        if near(x[1], 1.0):
            y[0] = x[0]
            y[1] = x[1] - 2.0

# Function space, test function, 
V = FunctionSpace(mesh, 'CG', 1, constrained_domain=PeriodicBoundary())
v = TestFunction(V)

# Define the functions to store the solution
u  = Function(V)
u1 = Function(V)
u0 = Function(V)
u_ = Function(V)
A_inv_b = Function(V)
A_inv_g = Function(V)

# Give initial condition
class u0_(UserExpression):
    def eval(self, values, x):
        if 4*x[0]**2+4*x[1]**2< 1:
            values[0] = 1
        # elif 16*(x[0]+0.5)**2+16*x[1]**2< 1:
        #     values[0] = 1
        # elif x[0]<0.5 and x[0]>-0.5 and x[1]<0.1 and x[1]>-0.1:
        #     values[0] = 1
        else:
            values[0] = -1
u0.interpolate(u0_())
u1.assign(u0)

# Define the boundary condition
# bc = DirichletBC(V, Constant(-1), bd)

# Compute r
r0 = sqrt(assemble(F(u0)*dx)+C0)
r1 = r0

# bilinear forms
a_u_ =  1/dt * u_ * v * dx\
    + epsilon *inner(grad(u_), grad(v))*dx\
    - 1/dt * u1 * v * dx - 1/epsilon * f(u_)*v*dx

# give initial guess to u_
u_.assign(u0)

# write the initial condition
u0.rename('u', 'u')
file.write(u0, 0)
t+=dt
u1.rename('u', 'u')
file.write(u1, t)
# Perform the SAV scheme
while t < T:
    # update the time
    t += dt
    # Compute u_ approximation of u
    solve(a_u_ == 0, u_,solver_parameters={"snes_solver" : {"maximum_iterations": 100, "absolute_tolerance": 1e-10, "relative_tolerance": 1e-9}})
    # u_ = 2*u1-u0
    # Compute gamma
    SAV_denominator = sqrt(assemble(F(u_)*dx)+C0)
    b = -f(u_)/SAV_denominator
    a_A_inv_b = A_inv_b*v*dx\
      + (2*dt)/(3)*epsilon*inner(grad(A_inv_b), grad(v))*dx\
      - b*v*dx
    solve(a_A_inv_b == 0, A_inv_b)
    gamma = assemble(b*A_inv_b*dx)
    # Compute the inner product
    inner_product_1 = assemble(b*(4*u1-u0)*dx)
    g = 4/3 * u1 - 1/3 * u0\
          - 2*dt/(9*epsilon) * (4*r1-r0 - 0.5*inner_product_1) * b
    a_A_inv_g = A_inv_g*v*dx\
      + (2*dt)/(3)*epsilon*inner(grad(A_inv_g), grad(v))*dx\
      - g*v*dx
    solve(a_A_inv_g == 0, A_inv_g)
    inner_product_2 = assemble(b*A_inv_g*dx)/(1+dt*gamma/(3*epsilon))
    # Compute the solution
    a = u*v*dx\
        +(2*dt)/(3)*epsilon*inner(grad(u), grad(v))*dx\
        - g*v*dx\
        + dt/(3*epsilon)*b*inner_product_2*v*dx
    solve(a == 0, u)
    # update the solution
    r = 4/3 * r1 - 1/3 * r0 + dt/3 * assemble(b*(3*u-4*u1+u0)/(2*dt)*dx)
    r0 = r1
    r1 = r
    u0.assign(u1)
    u1.assign(u)
    # store the solution
    u_.rename('u_', 'u_')
    u.rename('u', 'u')
    file.write(u, t)
    file.write(u_, t)
    info("----------------  time at {:.5f} / {:.5f}".format(t, T))
    S.append(assemble((u+Constant(1))/2*dx))
    t_axis.append(t)

# # compute the area inside the interface to check the solution
# print("Area of the disc")
# print(S)
# print("Time axis")
# print(t_axis)    