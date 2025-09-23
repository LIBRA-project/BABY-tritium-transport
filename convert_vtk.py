import dolfinx
from basix.ufl import element
from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pyvista
import ufl
from dolfinx.mesh import create_mesh


class VTUReader(pyvista.XMLUnstructuredGridReader):
    cell_type = "tetrahedron"
    connectivity: np.ndarray
    dolfinx_mesh: dolfinx.mesh.Mesh = None

    @property
    def cell_connectivity(self):
        return self._data.cells_dict[10]

    def create_dolfinx_mesh(self):
        """Creates the dolfinx mesh depending on the type of cell provided

        args:
            cell_type: the cell type for the dolfinx mesh, defaults to "tetrahedron"
        """

        # TODO find a way to fix this with abstractmethod and property
        if not hasattr(self, "cell_type"):
            raise AttributeError("cell_type must be defined in the child class")

        self._data = self.read()
        if not hasattr(self, "cell_connectivity"):
            raise AttributeError("cell_connectivity must be defined in the child class")

        degree = 1  # Set polynomial degree

        cell = ufl.Cell(f"{self.cell_type}")
        mesh_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=(3,)
        )

        # Create dolfinx Mesh
        mesh_ufl = ufl.Mesh(mesh_element)
        self.dolfinx_mesh = create_mesh(
            MPI.COMM_WORLD, self.cell_connectivity, self._data.points, mesh_ufl
        )

    def create_dolfinx_function(
        self, data: str = "mean", shape=(3,)
    ) -> dolfinx.fem.Function:
        """reads the filename of the OpenMC file

        Arguments:
            data: the name of the data to extract from the vtk file

        Returns:
            dolfinx function with openmc results mapped
        """

        if not self.dolfinx_mesh:
            self.create_dolfinx_mesh()

        el = element(
            "Lagrange",
            self.dolfinx_mesh.topology.cell_name(),
            degree=1,
            shape=shape,
        )

        function_space = dolfinx.fem.functionspace(self.dolfinx_mesh, el)
        u = dolfinx.fem.Function(function_space)

        num_vertices = (
            self.dolfinx_mesh.topology.index_map(0).size_local
            + self.dolfinx_mesh.topology.index_map(0).num_ghosts
        )
        vertex_map = np.empty(num_vertices, dtype=np.int32)

        # Get cell-to-vertex connectivity
        c_to_v = self.dolfinx_mesh.topology.connectivity(
            self.dolfinx_mesh.topology.dim, 0
        )
        # Map the OF_mesh vertices to dolfinx_mesh vertices
        num_cells = (
            self.dolfinx_mesh.topology.index_map(
                self.dolfinx_mesh.topology.dim
            ).size_local
            + self.dolfinx_mesh.topology.index_map(
                self.dolfinx_mesh.topology.dim
            ).num_ghosts
        )
        vertices = np.array([c_to_v.links(cell) for cell in range(num_cells)])
        flat_vertices = np.concatenate(vertices)
        cell_indices = np.repeat(np.arange(num_cells), [len(v) for v in vertices])
        vertex_positions = np.concatenate([np.arange(len(v)) for v in vertices])

        vertex_map[flat_vertices] = self.cell_connectivity[
            self.dolfinx_mesh.topology.original_cell_index
        ][cell_indices, vertex_positions]

        u.x.array[:] = self._data.point_data[f"{data}"][vertex_map]

        return u


if __name__ == "__main__":
    reader = VTUReader("OV Sweep Export.vtu")

    reader.create_dolfinx_mesh()

    # write mesh to VTX
    writer = dolfinx.io.VTXWriter(MPI.COMM_WORLD, "mesh.bp", reader.dolfinx_mesh)
    writer.write(0.0)

    # reader.create_dolfinx_function("U")
    u = reader.create_dolfinx_function("Velocity_field,_z-component", shape=None)

    # write mesh to VTX
    writer2 = dolfinx.io.VTXWriter(MPI.COMM_WORLD, "u.bp", u)
    writer2.write(0.0)
