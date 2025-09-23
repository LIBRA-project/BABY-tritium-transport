import festim as F
import numpy as np
from convert_vtk import VTUReader
import ufl

reader = VTUReader("OV Sweep Export.vtu")

reader.create_dolfinx_mesh()

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(reader.dolfinx_mesh)

gas = F.Material(name="Gas", D_0=1e-6, E_D=0)

gas_vol = F.VolumeSubdomain(id=1, material=gas)


def outgas_surf_loc(x):
    outer_radius = 0.073
    bottom_surf_z = 0.0125
    wetted_height = 0.07
    on_radius = np.isclose(x[0] ** 2 + x[1] ** 2, outer_radius**2, atol=1e-4)
    on_ring = np.logical_and(on_radius, x[2] <= wetted_height)
    on_bottom_surf = np.logical_and(
        np.isclose(x[2], bottom_surf_z, atol=1e-3),
        x[0] ** 2 + x[1] ** 2 <= outer_radius**2,
    )
    return np.logical_or(on_ring, on_bottom_surf)


def inlet_surf_loc(x):
    inlet_center_x = 0.049
    inlet_center_y = 0.085
    inlet_center_z = 0.208
    inlet_radius = 0.054 - inlet_center_x
    on_inlet = (x[0] - inlet_center_x) ** 2 + (
        x[1] - inlet_center_y
    ) ** 2 <= inlet_radius**2
    return np.logical_and(on_inlet, np.isclose(x[2], inlet_center_z, atol=1e-2))


outgassing_surf = F.SurfaceSubdomain(2, locator=outgas_surf_loc)
inlet_surf = F.SurfaceSubdomain(3, locator=inlet_surf_loc)

my_model.subdomains = [gas_vol, outgassing_surf, inlet_surf]

T = F.Species("T")

my_model.species = [T]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=inlet_surf, species=T, value=0.0),
    F.FixedConcentrationBC(
        subdomain=outgassing_surf,
        species=T,
        value=1.0,
    ),
]

# make vector function for velocity based on three components
print("Extracting velocity field from OV sweep export")
u_x = reader.create_dolfinx_function(data="Velocity_field,_x-component", shape=None)
u_y = reader.create_dolfinx_function(data="Velocity_field,_y-component", shape=None)
u_z = reader.create_dolfinx_function(data="Velocity_field,_z-component", shape=None)

print("Creating velocity field")
import dolfinx
from basix.ufl import element

el = element(
    "Lagrange",
    reader.dolfinx_mesh.topology.cell_name(),
    degree=1,
    shape=(3,),
)
V = dolfinx.fem.functionspace(mesh=reader.dolfinx_mesh, element=el)
velocity = dolfinx.fem.Function(V)
vx, vy, vz = ufl.split(velocity)

V_ux, V_ux_map = V.sub(0).collapse()
V_uy, V_uy_map = V.sub(1).collapse()
V_uz, V_uz_map = V.sub(2).collapse()
velocity.x.array[V_ux_map] = u_x.x.array[:]
velocity.x.array[V_uy_map] = u_y.x.array[:]
velocity.x.array[V_uz_map] = u_z.x.array[:]


my_model.advection_terms = [
    F.AdvectionTerm(velocity=velocity, species=T, subdomain=gas_vol),
]

my_model.temperature = 300
my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
    # final_time=10000,
)
# my_model.settings.stepsize = F.Stepsize(
#     initial_value=1000,
#     growth_factor=1.1,
#     cutback_factor=0.9,
#     target_nb_iterations=4,
# )

my_model.exports = [F.VTXSpeciesExport(filename="result.bp", field=T)]

my_model.initialise()
my_model.run()

# ft, ct = my_model.mesh.define_meshtags(
#     surface_subdomains=my_model.surface_subdomains,
#     volume_subdomains=my_model.volume_subdomains,
# )

# print(np.unique(ft.values))

# import pyvista
# from dolfinx import plot

# mesh = reader.dolfinx_mesh
# fdim = mesh.topology.dim - 1
# tdim = mesh.topology.dim
# mesh.topology.create_connectivity(fdim, tdim)
# topology, cell_types, x = plot.vtk_mesh(mesh, fdim, ft.indices)

# p = pyvista.Plotter()
# grid = pyvista.UnstructuredGrid(topology, cell_types, x)
# grid.cell_data["Facet Marker"] = ft.values
# grid.set_active_scalars("Facet Marker")

# # clip grid
# # grid = grid.clip(normal="x", origin=(0, 0, 0), invert=False)

# p.add_mesh(grid, show_edges=False)
# if pyvista.OFF_SCREEN:
#     figure = p.screenshot("facet_marker.png")
# p.show()
