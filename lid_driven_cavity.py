import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, io
from dolfinx.mesh import CellType, create_unit_square
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.log import set_log_level, LogLevel
import basix
import adios4dolfinx

# Create unit square mesh
domain = create_unit_square(MPI.COMM_WORLD, 32, 32, CellType.triangle)

# Define Taylor-Hood element (P2-P1)
element_u = basix.ufl.element(
    "Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,)
)
element_p = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
mixed_ele = basix.ufl.mixed_element([element_u, element_p])
VQ = fem.functionspace(domain, mixed_ele)

# Functions and test functions
up = fem.Function(VQ)
v_q = ufl.TestFunctions(VQ)
v, q = v_q
u, p = ufl.split(up)

V_collapsed, _ = VQ.sub(0).collapse()

# Viscosity
nu = 1e-03


# Lid velocity condition (top boundary)
def lid(x):
    return np.isclose(x[1], 1)


class LidVelocity:
    def __call__(self, x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = np.ones((1, x.shape[1]), dtype=PETSc.ScalarType) * 1e-4
        values[1] = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
        return values


lid_velocity = LidVelocity()

lid_u = fem.Function(V_collapsed)
lid_u.interpolate(lid_velocity)

facets_lid = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lid)
dofs_lid = fem.locate_dofs_topological((VQ.sub(0), V_collapsed), 1, facets_lid)
bc_lid = fem.dirichletbc(lid_u, dofs_lid, VQ.sub(0))


# No-slip walls
def walls(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)


wall_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, walls)
wall_dofs = fem.locate_dofs_topological((VQ.sub(0), V_collapsed), 1, wall_facets)

u_noslip = fem.Function(V_collapsed)
u_noslip.x.array[:] = 0.0
bc_walls = fem.dirichletbc(u_noslip, wall_dofs, VQ.sub(0))

bcs = [bc_lid, bc_walls]


# Define variational problem (steady Navier-Stokes)
F = (
    nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
    - p * ufl.div(v) * ufl.dx
    - q * ufl.div(u) * ufl.dx
)

J = ufl.derivative(F, up)

set_log_level(LogLevel.INFO)

# Create nonlinear problem and solver
problem = NonlinearProblem(F, up, bcs, J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.atol = 1e-10
solver.rtol = 1e-8
solver.max_it = 50
solver.report = True
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# Solve
n, converged = solver.solve(up)
print(f"Converged: {converged} in {n} Newton iterations")

# Extract solutions
uh, ph = up.sub(0).collapse(), up.sub(1).collapse()
uh.name = "velocity"
ph.name = "pressure"

writer = io.VTXWriter(MPI.COMM_WORLD, "lid_driven_cavity.bp", uh, "BP5")
writer.write(t=0)

adios4dolfinx.write_mesh("lid_driven_cavity_cp.bp", mesh=domain)
adios4dolfinx.write_function("lid_driven_cavity_cp.bp", uh, time=0.0)
