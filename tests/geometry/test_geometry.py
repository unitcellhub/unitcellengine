import unittest
import unitcellengine.geometry as geometry
from pathlib import Path
import json

# class TestNtop(unittest.TestCase):

#     def text_bcc(self):
#         """ Test the generation of a body centered cubic lattice """
#         lattice = ntop.LatticeUnitcell('Body centered cubic', 1, 1, 1,
#                                         directory=Path(__file__) / Path('ntop'))


class TestSDFGeometry(unittest.TestCase):
    """ Test elastic homogenization method """

    def testSimpleCubicFoam(self):
        """ Compare geometry hand calcs with output values """

        # Define geometry object
        L = 1
        t = 0.15
        directory = Path(__file__).parent / Path('resources')
        lattice = geometry.sdf.SDFGeometry('Simple cubic foam', 
                                            L, L, L,
                                            thickness=t,
                                            directory=directory)

        # Generate the geometry and process the results
        lattice.run(reuse=False)
        
        # Check that the necessary output files are present
        self.assertTrue(lattice.definitionFilename.exists())
        with lattice.definitionFilename.open('r') as f:
            definition = json.load(f)
        self.assertListEqual(list(definition.keys()), 
                             ["unitcell", "length", "width", "height",
                              "thickness", "radius", "elementSize", "form"])
        self.assertTrue(lattice.propertiesFilename.exists())
        with lattice.propertiesFilename.open('r') as f:
            properties = json.load(f)
        self.assertListEqual(list(properties.keys()), 
                             ["relativeDensity", "relativeSurfaceArea"])

        # Check relative density
        expDensity = (L*L*t+L*(L-t)*t+(L-t)**2*t)/(L**3)
        self.assertAlmostEqual(lattice.relativeDensity, expDensity, 2)

        # Check exposed internal surface area
        expArea = 2*3*((L-t)**2)/(6*L*L)
        self.assertAlmostEqual(lattice.relativeSurfaceArea, expArea, 1)

if __name__ == "__main__":
    unittest.main()
