from foam2dolfinx import OpenFOAMReader

from pathlib import Path

filename = Path("../openfoam/pv.foam")

reader = OpenFOAMReader(filename, cell_type=10)
vel = reader.create_dolfinx_function(t=870, name="U")
