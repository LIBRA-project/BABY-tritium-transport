import festim as F
from openmc2dolfinx import UnstructuredMeshReader


# NOTE need to override these methods in ParticleSource until a
# new version of festim is released
class ParticleSourceFromOpenMC(F.ParticleSource):
    @property
    def temperature_dependent(self):
        return False

    @property
    def time_dependent(self):
        return False


# convert openmc result
reader = UnstructuredMeshReader("um_tbr.vtk")
reader.create_dolfinx_mesh()


dolfinx_mesh = reader.dolfinx_mesh
dolfinx_mesh.geometry.x[:] /= 100

tritium_source_term = reader.create_dolfinx_function("mean")

neutron_rate = 1e8  # n/s
tritium_source_term.x.array[:] *= neutron_rate

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(mesh=reader.dolfinx_mesh)

T = F.Species("T")
my_model.species = [T]

salt = F.Material(D_0=1, E_D=0)
volume = F.VolumeSubdomain(id=1, material=salt)

my_model.subdomains = [volume]

my_model.sources = [
    ParticleSourceFromOpenMC(value=tritium_source_term, volume=volume, species=T),
]

my_model.temperature = 650 + 273.15  # 650 degC

my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    final_time=3600,
    stepsize=100,
)

my_model.exports = [
    F.VTXSpeciesExport("tritium_concentration.bp", field=T),
]

# from dolfinx.log import set_log_level, LogLevel

# set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()
