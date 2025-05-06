import festim as F
from openmc2dolfinx import UnstructuredMeshReader
import dolfinx
import numpy as np
import matplotlib.pyplot as plt
from libra_toolbox.tritium.model import quantity_to_activity, ureg
from scipy.integrate import cumulative_trapezoid
import requests
from dolfinx import fem
import basix
from foam2dolfinx import OpenFOAMReader
import ufl
from mpi4py import MPI


# NOTE need to override these methods in ParticleSource until a
# new version of festim is released
class ValueFromOpenMC(F.Value):
    explicit_time_dependent = True
    temperature_dependent = False

    def update(self, t):
        if t < irradiation_time:
            return
        else:
            self.fenics_object.x.array[:] = 0.0


class SourceFromOpenMC(F.ParticleSource):
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if isinstance(value, F.Value):
            self._value = value
        else:
            self._value = F.Value(value)


def read_openmc_data(wedge_mesh):
    """Read OpenMC data from a file."""
    reader = UnstructuredMeshReader("um_tbr_rem.vtk")
    openmc_full = reader.create_dolfinx_function(data="mean")

    full_mesh = openmc_full.function_space.mesh
    # convert openmc mesh from cm to meters
    full_mesh.geometry.x[:] /= 100

    # translate the points to 0,0,0 origin
    full_mesh.geometry.x[:] -= np.array([5.87, 0.6, 1.06766])

    degree = 1  # Set polynomial degree
    cell = ufl.Cell("tetrahedron")
    element = basix.ufl.element("Lagrange", cell.cellname(), degree, shape=())
    V = fem.functionspace(wedge_mesh, element)
    openmc_wedge = fem.Function(V)

    F.helpers.nmm_interpolate(openmc_wedge, openmc_full)

    return openmc_wedge


# read OpenFOAM data
my_of_reader = OpenFOAMReader(filename="foam_data/pv.foam", cell_type=10)
u_wedge = my_of_reader.create_dolfinx_function(t=870.0, name="U")
# option to write velocity field to file to check
# writer = dolfinx.io.VTXWriter(MPI.COMM_WORLD, "velocity.bp", u_wedge, "BP5")
# writer.write(t=0)


irradiation_time = 12 * 3600  # 12 hours

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.MeshFromXDMF(
    volume_file="mesh/mesh_domains.xdmf", facet_file="mesh/mesh_boundaries.xdmf"
)


T = F.Species("T")
my_model.species = [T]

flibe_salt = F.Material(D_0=3.12e-7, E_D=0.37)
flinak_salt = F.Material(D_0=4.01e-7, E_D=0.31)
salt = flibe_salt
volume = F.VolumeSubdomain(id=6, material=salt)
top_boundary = F.SurfaceSubdomain(id=7)
my_model.subdomains = [volume, top_boundary]


tritium_source_term = read_openmc_data(my_model.mesh.mesh)
neutron_rate = 1.2e8  # n/s
percm3_to_perm3 = 1e6
# convert source term from T/n/cm3 to T/s/cm3
tritium_source_term.x.array[:] *= neutron_rate
# convert source term from T/s/cm3 to T/s/m3
tritium_source_term.x.array[:] *= percm3_to_perm3
# option to write source term to file to check
# writer = dolfinx.io.VTXWriter(
#     MPI.COMM_WORLD, "tritium_source.bp", tritium_source_term, "BP5"
# )
# writer.write(t=0)
my_model.sources = [
    SourceFromOpenMC(
        value=ValueFromOpenMC(tritium_source_term), volume=volume, species=T
    ),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_boundary, species=T, value=0)
]

my_model.temperature = 650 + 273.15  # 650 degC

my_advection = F.AdvectionTerm(
    velocity=u_wedge,
    subdomain=volume,
    species=T,
)
my_model.advection_terms = [my_advection]

dt = F.Stepsize(
    3600,
    growth_factor=1.2,
    cutback_factor=0.9,
    target_nb_iterations=4,
    milestones=[irradiation_time],
)

my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-20,
    final_time=60 * 24 * 3600,
    stepsize=dt,
)

top_release = F.SurfaceFlux(field=T, surface=top_boundary)
inv = F.TotalVolume(field=T, volume=volume)

my_model.exports = [
    F.VTXSpeciesExport("tritium_concentration_with_advection.bp", field=T),
    top_release,
    inv,
]

# set_log_level(LogLevel.INFO)

my_model.initialise()

# option to write velocity field to file to check
# writer = dolfinx.io.VTXWriter(
#     MPI.COMM_WORLD, "velocity_festim.bp", my_advection.velocity.fenics_object, "BP5"
# )
# writer.write(t=0)

my_model.run()

# Plot results
wedge_angle = 15  # degrees  according to Online Protractor
top_release_full_surface = np.array(top_release.data) * 360 / wedge_angle


import morethemes as mt

mt.set_theme("minimal")
s_to_day = 1 / 3600 / 24

fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

axs[0].plot(np.array(top_release.t) * s_to_day, top_release_full_surface)
axs[0].set_ylabel("Tritium flux [T s-1]")

# compute cumulative release as int(release dt)

cumulative_release = cumulative_trapezoid(
    top_release_full_surface, x=top_release.t, initial=0
)

# convert to Bq
cumulative_release = (
    quantity_to_activity(ureg.Quantity(cumulative_release, "particle"))
    .to(ureg.Bq)
    .magnitude
)

axs[1].plot(np.array(top_release.t) * s_to_day, cumulative_release, label="FESTIM")
axs[1].set_ylabel("Cumulative tritium release [Bq]")
plt.xlabel("Time [day]")


# read experimental data
url = "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.5/data/processed_data.json"

experimental_data = requests.get(url).json()

cumulative_release_exp = experimental_data["cumulative_tritium_release"]["IV"]["total"][
    "value"
]
sampling_times = experimental_data["cumulative_tritium_release"]["IV"][
    "sampling_times"
]["value"]

axs[1].scatter(sampling_times, cumulative_release_exp, label="Experiment")

axs[1].legend()
axs[1].set_ylim(bottom=0)

plt.sca(axs[1])
plt.axvspan(0, irradiation_time * s_to_day, facecolor="#EF5B5B", alpha=0.5)


plt.show()
