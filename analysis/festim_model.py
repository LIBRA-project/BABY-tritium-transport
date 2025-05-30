import festim as F
from dolfinx.log import set_log_level, LogLevel
from libra_toolbox.tritium.model import ureg
import requests
import h_transport_materials as htm
import ufl
from dolfinx import fem

id_flibe = 6
id_inconel = 7

id_flibe_top_surface = 8

id_inconel_inner_side_surface = 9
id_inconel_inner_top_surface = 10
id_inconel_outer_bottom_surface = 12
id_inconel_outer_side_surface = 13
id_inconel_outer_top_surface = 14

id_salt_metal_interface = 11


class MyParticleFluxBC(F.FluxBCBase):
    def __init__(
        self, subdomain, value, species, volume_subdomain, species_dependent_value={}
    ):
        super().__init__(subdomain=subdomain, value=value)
        self.species = species
        self.species_dependent_value = species_dependent_value
        self.volume_subdomain = volume_subdomain

    def create_value_fenics(self, mesh, temperature, t: fem.Constant):
        """Creates the value of the boundary condition as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a ufl Expression

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            temperature (float): the temperature
            t (dolfinx.fem.Constant): the time
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.value_fenics = F.as_fenics_constant(mesh=mesh, value=self.value)

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames

            if "t" in arguments and "x" not in arguments and "T" not in arguments:
                # only t is an argument
                if not isinstance(self.value(t=float(t)), (float, int)):
                    raise ValueError(
                        f"self.value should return a float or an int, not {type(self.value(t=float(t)))} "
                    )
                self.value_fenics = F.as_fenics_constant(
                    mesh=mesh, value=self.value(t=float(t))
                )
            else:
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x
                if "T" in arguments:
                    kwargs["T"] = temperature

                for name, species in self.species_dependent_value.items():
                    kwargs[name] = species.subdomain_to_solution[self.volume_subdomain]

                self.value_fenics = self.value(**kwargs)


class SurfaceFluxFromGradient(F.SurfaceQuantity):
    def __init__(self, field, surface, filename, volume_subdomain):
        super().__init__(field=field, surface=surface, filename=filename)
        self.volume_subdomain = volume_subdomain

    @property
    def title(self):
        return f"{self.field.name} flux surface {self.surface.id}"

    def compute(self, u, ds, entity_maps):
        """Computes the value of the flux at the surface

        Args:
            ds (ufl.Measure): surface measure of the model
        """

        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh
        n = ufl.FacetNormal(mesh)

        self.value = fem.assemble_scalar(
            fem.form(
                -self.D * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        self.data.append(self.value)


# ##### get irradiation time from experimental data ##### #


def get_total_irradiation_time() -> float:
    # read experimental data
    url = "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.5/data/processed_data.json"

    experimental_data = requests.get(url).json()

    duration = 0
    irradiations = experimental_data["irradiations"]
    for irr in irradiations:
        start = irr["start_time"]["value"] * ureg(irr["start_time"]["unit"])
        end = irr["stop_time"]["value"] * ureg(irr["stop_time"]["unit"])
        duration += (end - start).to(ureg("s")).magnitude
    return duration


irradiation_time = get_total_irradiation_time()

# ##### get material properties ##### #

htm_D_flibe = htm.diffusivities.filter(material="flibe").filter(author="calderoni")
htm_S_flibe = htm.solubilities.filter(material="flibe").filter(author="calderoni")

flibe_D_0 = htm_D_flibe[0].pre_exp.magnitude
flibe_E_D = htm_D_flibe[0].act_energy.magnitude
flibe_S_0 = htm_S_flibe[0].pre_exp.magnitude
flibe_E_S = htm_S_flibe[0].act_energy.magnitude

htm_D_inconel = htm.diffusivities.filter(material="inconel_625")
htm_S_inconel = htm.solubilities.filter(material="inconel_625")
htm_recomb_inconel = htm.recombination_coeffs.filter(material="inconel_625")

inconel_D_0 = htm_D_inconel[0].pre_exp.magnitude
inconel_E_D = htm_D_inconel[0].act_energy.magnitude
inconel_S_0 = htm_S_inconel[0].pre_exp.magnitude
inconel_E_S = htm_S_inconel[0].act_energy.magnitude

inconel_Kr_0 = htm_recomb_inconel[1].pre_exp.magnitude
inconel_E_Kr = htm_recomb_inconel[1].act_energy.magnitude


flibe_D_0 *= 1

# ##### create festim models ##### #


class SurfaceFluxFromEquation(F.SurfaceQuantity):
    def __init__(self, field, surface, filename, volume_subdomain):
        super().__init__(field=field, surface=surface, filename=filename)
        self.volume_subdomain = volume_subdomain

    @property
    def title(self):
        return f"{self.field.name} calculated flux surface {self.surface.id}"

    def compute(self, u, ds, entity_maps):
        """Computes the value of the flux at the surface

        Args:
            ds (ufl.Measure): surface measure of the model
        """
        temperature = 650 + 273.15  # 650 degC
        Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * temperature))
        self.value = fem.assemble_scalar(
            fem.form(
                -Kr * u**2 * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        self.data.append(self.value)


def my_source(t):
    if t < irradiation_time:
        return 2.19e8
    else:
        return 0.0


my_mesh = F.MeshFromXDMF(
    volume_file="mesh/mesh_domains.xdmf", facet_file="mesh/mesh_boundaries.xdmf"
)

mat_flibe = F.Material(
    D_0=flibe_D_0,
    E_D=flibe_E_D,
    K_S_0=flibe_S_0,
    E_K_S=flibe_E_S,
    solubility_law="henry",
)
mat_inconel = F.Material(
    D_0=inconel_D_0,
    E_D=inconel_E_D,
    K_S_0=inconel_S_0,
    E_K_S=inconel_E_S,
    solubility_law="sievert",
)

vol_flibe = F.VolumeSubdomain(id=id_flibe, material=mat_flibe)
vol_inconel = F.VolumeSubdomain(id=id_inconel, material=mat_inconel)

flibe_top_surface = F.SurfaceSubdomain(id=id_flibe_top_surface)
inconel_inner_side = F.SurfaceSubdomain(id=id_inconel_inner_side_surface)
inconel_inner_top = F.SurfaceSubdomain(id=id_inconel_inner_top_surface)
inconel_outer_bottom = F.SurfaceSubdomain(id=id_inconel_outer_bottom_surface)
inconel_outer_side = F.SurfaceSubdomain(id=id_inconel_outer_side_surface)
inconel_outer_top = F.SurfaceSubdomain(id=id_inconel_outer_top_surface)

salt_metal_interface = F.Interface(
    id=id_salt_metal_interface,
    subdomains=[vol_flibe, vol_inconel],
    penalty_term=1e23,
)

dt = F.Stepsize(
    1,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
    milestones=[irradiation_time],
)

my_settings = F.Settings(
    transient=True,
    atol=1e-12,
    rtol=1e-16,
    final_time=60 * 24 * 3600,
    stepsize=dt,
)


def festim_model(h2: bool):
    my_model = F.HydrogenTransportProblemDiscontinuous()

    my_model.mesh = my_mesh

    my_model.subdomains = [
        vol_inconel,
        vol_flibe,
        flibe_top_surface,
        inconel_inner_side,
        inconel_inner_top,
        inconel_outer_bottom,
        inconel_outer_side,
        inconel_outer_top,
    ]

    my_model.interfaces = [salt_metal_interface]
    my_model.surface_to_volume = {
        flibe_top_surface: vol_flibe,
        inconel_inner_side: vol_inconel,
        inconel_inner_top: vol_inconel,
        inconel_outer_bottom: vol_inconel,
        inconel_outer_side: vol_inconel,
        inconel_outer_top: vol_inconel,
    }

    T = F.Species("T", mobile=True, subdomains=[vol_flibe, vol_inconel])
    my_model.species = [T]

    my_model.sources = [
        F.ParticleSource(value=my_source, volume=vol_flibe, species=T),
    ]

    if h2:
        h2_P_gauge = 3  # psi
        h2_conc_ppm = 1000  # ppm

        mole_frac_h2 = h2_conc_ppm / 1e6  # ppm to mole fraction
        P_atm = 14.7  # psi
        P_abs = h2_P_gauge + P_atm

        P_h2 = mole_frac_h2 * P_abs  # atm
        P_h2 *= 6894.76  # convert psi to Pa
        gas_constant = 8.314
        temperature = 298  # K
        h2_conc_mol = P_h2 / (gas_constant * temperature)  # mol/m3
        h2_conc = h2_conc_mol * 6.022e23  # convert mol/m3 to m-3

        recombination_flux = (
            lambda c, T: -(inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))) * c**2
            - (inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))) * h2_conc * c
        )
        folder = "results/h2"

    else:
        recombination_flux = (
            lambda c, T: -(inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))) * c**2
        )
        folder = "results/he"

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=flibe_top_surface, species=T, value=0.0),
        # MyParticleFluxBC(
        #     value=recombination_flux,
        #     subdomain=inconel_inner_side,
        #     species_dependent_value={"c": T},
        #     species=T,
        #     volume_subdomain=vol_inconel,
        # ),
        # MyParticleFluxBC(
        #     value=recombination_flux,
        #     subdomain=inconel_inner_top,
        #     species_dependent_value={"c": T},
        #     species=T,
        #     volume_subdomain=vol_inconel,
        # ),
        # MyParticleFluxBC(
        #     value=recombination_flux,
        #     subdomain=inconel_outer_bottom,
        #     species_dependent_value={"c": T},
        #     species=T,
        #     volume_subdomain=vol_inconel,
        # ),
        # MyParticleFluxBC(
        #     value=recombination_flux,
        #     subdomain=inconel_outer_side,
        #     species_dependent_value={"c": T},
        #     species=T,
        #     volume_subdomain=vol_inconel,
        # ),
        # MyParticleFluxBC(
        #     value=recombination_flux,
        #     subdomain=inconel_outer_top,
        #     species_dependent_value={"c": T},
        #     species=T,
        #     volume_subdomain=vol_inconel,
        # ),
    ]

    my_model.temperature = 650 + 273.15  # 650 degC

    my_model.settings = my_settings

    my_model.exports = [
        F.VTXSpeciesExport(
            filename=f"{folder}/tritium_concentration_flibe.bp",
            field=T,
            subdomain=vol_flibe,
        ),
        F.VTXSpeciesExport(
            filename=f"{folder}/tritium_concentration_inconel.bp",
            field=T,
            subdomain=vol_inconel,
        ),
        SurfaceFluxFromGradient(
            field=T,
            surface=flibe_top_surface,
            filename=f"{folder}/surface_flux_flibe_top.csv",
            volume_subdomain=vol_flibe,
        ),
        SurfaceFluxFromGradient(
            field=T,
            surface=inconel_inner_side,
            filename=f"{folder}/surface_flux_inconel_inner_side.csv",
            volume_subdomain=vol_inconel,
        ),
        SurfaceFluxFromGradient(
            field=T,
            surface=inconel_inner_top,
            filename=f"{folder}/surface_flux_inconel_inner_top.csv",
            volume_subdomain=vol_inconel,
        ),
        SurfaceFluxFromGradient(
            field=T,
            surface=inconel_outer_bottom,
            filename=f"{folder}/surface_flux_inconel_outer_bottom.csv",
            volume_subdomain=vol_inconel,
        ),
        SurfaceFluxFromGradient(
            field=T,
            surface=inconel_outer_side,
            filename=f"{folder}/surface_flux_inconel_outer_side.csv",
            volume_subdomain=vol_inconel,
        ),
        SurfaceFluxFromGradient(
            field=T,
            surface=inconel_outer_top,
            filename=f"{folder}/surface_flux_inconel_outer_top.csv",
            volume_subdomain=vol_inconel,
        ),
        SurfaceFluxFromEquation(
            field=T,
            surface=inconel_outer_bottom,
            filename=f"{folder}/surface_flux_inconel_outer_bottom_calculated.csv",
            volume_subdomain=vol_inconel,
        ),
        SurfaceFluxFromEquation(
            field=T,
            surface=inconel_outer_side,
            filename=f"{folder}/surface_flux_inconel_outer_side_calculated.csv",
            volume_subdomain=vol_inconel,
        ),
    ]

    return my_model


if __name__ == "__main__":
    set_log_level(LogLevel.INFO)

    model = festim_model(h2=False)
    model.initialise()
    model.run()
