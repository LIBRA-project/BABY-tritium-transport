import festim as F
import dolfinx
import numpy as np
from dolfinx import fem
import basix
import adios4dolfinx
from mpi4py import MPI


def festim_sim(vel_factor, folder):
    my_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)

    # read velocity field
    mesh_ldc = adios4dolfinx.read_mesh("lid_driven_cavity_cp.bp", MPI.COMM_WORLD)
    P2 = basix.ufl.element(
        "Lagrange", mesh_ldc.basix_cell(), 2, shape=(mesh_ldc.geometry.dim,)
    )
    V = dolfinx.fem.functionspace(mesh_ldc, P2)
    u_ldc_original = dolfinx.fem.Function(V)
    u_ldc_original.name = "velocity"
    adios4dolfinx.read_function("lid_driven_cavity_cp.bp", u_ldc_original, time=0.0)

    ele = basix.ufl.element("P", my_mesh.basix_cell(), 1, shape=(3,))
    V = fem.functionspace(my_mesh, ele)
    u_ldc_festim = fem.Function(V)
    F.helpers.nmm_interpolate(u_ldc_festim, u_ldc_original)
    u_ldc_festim.x.array[:] *= vel_factor

    my_model = F.HydrogenTransportProblem()

    my_model.mesh = F.Mesh(mesh=my_mesh)

    class TopSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = dolfinx.mesh.locate_entities_boundary(
                mesh, fdim, lambda x: np.isclose(x[1], 1)
            )
            return indices

    T = F.Species("T")
    my_model.species = [T]

    volume = F.VolumeSubdomain(id=1, material=F.Material(D_0=1e-4, E_D=0.1))
    top_boundary = TopSurface(id=2)

    my_model.subdomains = [volume, top_boundary]

    irradiation_time = 100
    my_model.sources = [
        F.ParticleSource(
            value=lambda t: 100 if t <= irradiation_time else 0,
            volume=volume,
            species=T,
        ),
    ]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=top_boundary, species=T, value=0.0)
    ]

    my_model.temperature = 650 + 273.15  # 650 degC

    my_model.advection_terms = [
        F.AdvectionTerm(
            velocity=u_ldc_festim,
            subdomain=volume,
            species=T,
        )
    ]

    dt = F.Stepsize(
        1,
        growth_factor=1.2,
        cutback_factor=0.9,
        target_nb_iterations=4,
        milestones=[irradiation_time],
    )

    my_model.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        final_time=1e5,
        stepsize=dt,
    )

    top_release = F.SurfaceFlux(
        field=T, surface=top_boundary, filename=f"{folder}/top_release.csv"
    )
    inv = F.TotalVolume(field=T, volume=volume)

    my_model.exports = [
        F.VTXSpeciesExport(f"{folder}/test_ldc.bp", field=T),
        top_release,
        inv,
    ]

    # set_log_level(LogLevel.INFO)

    my_model.initialise()

    my_model.run()


values = np.linspace(100, 1000, num=6)

if __name__ == "__main__":
    for value in values:
        print("Running with vel factor", value)
        festim_sim(value, f"testing/vel_factor_{value:.1f}")
