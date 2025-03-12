import festim as F
from openmc2dolfinx import UnstructuredMeshReader
import dolfinx
from dolfinx.log import set_log_level, LogLevel
import numpy as np
import matplotlib.pyplot as plt
from libra_toolbox.tritium.model import quantity_to_activity, ureg
from scipy.integrate import cumulative_trapezoid
import requests


irradiation_time = 12 * 3600  # 12 hours


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
        if t < irradiation_time:
            return
        else:
            self.value_fenics.x.array[:] = 0.0


# convert openmc result
reader = UnstructuredMeshReader("um_tbr.vtk")
reader.create_dolfinx_mesh()


dolfinx_mesh = reader.dolfinx_mesh
dolfinx_mesh.geometry.x[:] /= 100

tritium_source_term = reader.create_dolfinx_function("mean")

neutron_rate = 1.2e8  # n/s
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

# salt = F.Material(D_0=0.6e-6, E_D=0.42)
salt = F.Material(D_0=6.75e-6, E_D=0.42)
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

top_release = F.SurfaceFlux(field=T, surface=top_boundary)

my_model.exports = [
    F.VTXSpeciesExport("tritium_concentration.bp", field=T),
    top_release,
]

# set_log_level(LogLevel.INFO)

my_model.initialise()

my_model.run()


s_to_day = 1 / 3600 / 24

fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

axs[0].plot(np.array(top_release.t) * s_to_day, top_release.data)
axs[0].set_ylabel("Tritium flux [T s-1]")

# compute cumulative release as int(release dt)

cumulative_release = cumulative_trapezoid(top_release.data, x=top_release.t, initial=0)

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
url = "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.4/data/processed_data.json"

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
