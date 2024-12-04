import json
import unittest
from pathlib import Path

import numpy as np

import unitcellengine.geometry.sdf as sdf

# class TestNtop(unittest.TestCase):

#     def text_bcc(self):
#         """ Test the generation of a body centered cubic lattice """
#         lattice = ntop.LatticeUnitcell('Body centered cubic', 1, 1, 1,
#                                         directory=Path(__file__) / Path('ntop'))


class TestSDFGeometry(unittest.TestCase):
    """Test elastic homogenization method"""

    def testSimpleCubicFoam(self):
        """Compare geometry hand calcs with output values"""

        # Define geometry object
        L = 1
        t = 0.15
        directory = Path(__file__).parent / Path("resources")
        lattice = sdf.SDFGeometry(
            "Simple cubic foam", L, L, L, thickness=t, directory=directory
        )

        # Generate the geometry and process the results
        lattice.run(reuse=False)

        # Check that the necessary output files are present
        self.assertTrue(lattice.definitionFilename.exists())
        with lattice.definitionFilename.open("r") as f:
            definition = json.load(f)
        self.assertListEqual(
            list(definition.keys()),
            [
                "unitcell",
                "length",
                "width",
                "height",
                "thickness",
                "radius",
                "elementSize",
                "form",
            ],
        )
        self.assertTrue(lattice.propertiesFilename.exists())
        with lattice.propertiesFilename.open("r") as f:
            properties = json.load(f)
        self.assertListEqual(
            list(properties.keys()), ["relativeDensity", "relativeSurfaceArea"]
        )

        # Check relative density
        expDensity = (L * L * t + L * (L - t) * t + (L - t) ** 2 * t) / (L**3)
        self.assertAlmostEqual(lattice.relativeDensity, expDensity, 2)

        # Check exposed internal surface area
        expArea = 2 * 3 * ((L - t) ** 2) / (6 * L * L)
        self.assertAlmostEqual(lattice.relativeSurfaceArea, expArea, 1)

    def testCustomGraph(self):
        """Compare custom simple cubic definition with built in definition"""

        # Define an output directory
        directory = Path(__file__).parent / Path("resources")

        # Define a simple cubic template geometry
        template = {
            "node": np.array(
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ]
            ),
            "beam": np.array(
                [
                    [0, 1],
                    [0, 2],
                    [2, 3],
                    [1, 3],
                    [4, 5],
                    [4, 6],
                    [6, 7],
                    [5, 7],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ]
            ),
            "face": {},
        }

        geometry = dict(
            length=1,
            width=1.5,
            height=1.25,
            thickness=0.3,
            radius=0.1,
            elementSize=0.5,
            form="graph",
            directory=directory,
        )
        builtin = sdf.SDFGeometry("Simple cubic", **geometry)
        custom = sdf.SDFGeometry(
            dict(name="Custom Simple Cubic", definition=template), **geometry
        )
        builtin.run(reuse=False, export=True)
        custom.run(reuse=False, export=True)

        # Check relative density and relative surface areas
        self.assertAlmostEqual(builtin.relativeDensity, custom.relativeDensity, 4)
        self.assertAlmostEqual(
            builtin.relativeSurfaceArea, custom.relativeSurfaceArea, 4
        )


if __name__ == "__main__":
    unittest.main()
