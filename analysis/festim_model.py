import festim as F
from openmc2dolfinx import UnstructuredMeshReader
import dolfinx
import numpy as np


# NOTE need to override these methods in ParticleSource until a
# new version of festim is released
class ParticleSourceFromOpenMC(F.ParticleSource):
    @property
    def temperature_dependent(self):
        return False

    @property
    def time_dependent(self):
        return True

    def update(self, t):
        if t < 12 * 3600:
            return
        else:
            self.value_fenics.x.array[:] = 0.0


# convert openmc result
reader = UnstructuredMeshReader("um_tbr.vtk")
reader.create_dolfinx_mesh()


dolfinx_mesh = reader.dolfinx_mesh
dolfinx_mesh.geometry.x[:] /= 100

tritium_source_term = reader.create_dolfinx_function("mean")

neutron_rate = 1e8  # n/s
percm3_to_perm3 = 1e6
tritium_source_term.x.array[:] *= neutron_rate * percm3_to_perm3

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(mesh=reader.dolfinx_mesh)


class TopSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        z_top = mesh.geometry.x[:, 2].max()
        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[2], z_top)
        )


T = F.Species("T")
my_model.species = [T]

salt = F.Material(D_0=1e-8, E_D=0.42)
volume = F.VolumeSubdomain(id=1, material=salt)
top_boundary = TopSurface(id=2)

my_model.subdomains = [volume, top_boundary]

my_model.sources = [
    ParticleSourceFromOpenMC(value=tritium_source_term, volume=volume, species=T),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_boundary, species=T, value=0.0)
]

my_model.temperature = 650 + 273.15  # 650 degC

dt = F.Stepsize(
    3600,
    growth_factor=1.2,
    cutback_factor=0.9,
    target_nb_iterations=4,
    milestones=[12 * 3600],
)

my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-20,
    final_time=60 * 24 * 3600,
    stepsize=dt,
)

my_model.exports = [
    F.VTXSpeciesExport("tritium_concentration.bp", field=T),
]

from dolfinx.log import set_log_level, LogLevel

# set_log_level(LogLevel.INFO)

my_model.initialise()

my_model.run()
