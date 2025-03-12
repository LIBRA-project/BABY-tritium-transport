import festim as F
from openmc2dolfinx import UnstructuredMeshReader

# convert openmc result
reader = UnstructuredMeshReader("um_tbr.vtk")
reader.create_dolfinx_mesh()

# TODO convert mesh from cm to m

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(mesh=reader.dolfinx_mesh)

T = F.Species("T")
my_model.species = [T]

salt = F.Material(D_0=1, E_D=0)
volume = F.VolumeSubdomain(id=1, material=salt)

my_model.subdomains = [volume]

my_model.sources = [
    F.ParticleSource(value=1, volume=volume, species=T),
]

my_model.temperature = 650 + 273.15

my_model.settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    final_time=3600,
    stepsize=10,
)

my_model.initialise()
my_model.run()
