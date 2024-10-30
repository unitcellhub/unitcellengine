import unittest
from unitcellengine.design import UnitcellDesign
from pathlib import Path
import numpy as np

class TestUnitcell(unittest.TestCase):
    """ Test unitcellengine definition and quantification """

    def testMeshConvergence(self):
        """ Verify that default mesh is sufficiently converged """

        # Create default geometry to pull out the reference point
        defaultElementSize = 0.25
        design = unitcellengine.UnitcellDesign('Octet', 1, 1, 1)
        design.generateGeometry(reuse=False)
        design.generateMesh(elementSize=defaultElementSize,
                            reuse=False)
        design.homogenizationElastic.run(blocking=True, reuse=False)
        check = design.homogenizationElastic.check()
        design.homogenizationElastic.process(check=True, reuse=False)

        
        defaultC = design.homogenizationElastic.CH.copy()
        # defaultMaxStresses = design.homogenizationElastic.maxStresses 

        # Check what happens when the element size is cut in half
        refinedElementSize =defaultElementSize*0.5 
        design.generateMesh(elementSize=refinedElementSize,
                            reuse=False)
        design.homogenizationElastic.run(blocking=True, reuse=False)
        check = design.homogenizationElastic.check()
        design.homogenizationElastic.process(check=True, reuse=False)

        refinedC = design.homogenizationElastic.CH.copy()
        # refinedMaxStresses = design.homogenizationElastic.maxStresses 

        # Check stiffness convergence
        atolC = refinedC.max()*1e-4
        self.assertTrue(np.allclose(defaultC, refinedC, atol=atolC))

        # # Check stress convergences
        # atolStress = refinedMaxStresses.max()*1e-4
        # self.assertTrue(np.allclose(defaultMaxStresses, refinedMaxStresses, 
        #                                  atol=atolStress))
        


        # # Loop through and very the element size in the mesh
        # elementSizes = [defaultElementSize*x for x in [4, 2, 0.5, 0.25]]
        # Cs = []
        # maxStresses = []
        # for elementSize in elementSizes:
        #     # Update the element size and reprocesses
        #     design.elementSize = elementSize
        #     design.generateMesh(check=True, reuse=False)
        #     design.homogenizationElastic.run(blocking=True, reuse=False)
        #     check = design.homogenizationElastic.check()
        #     design.homogenizationElastic.process(check=True)

        #     # Store results
        #     Cs.append(design.result)
        #     maxStresses.append(design.maxStress)
        
        # # Calculate differentials to check for convergence
        # tmpCs = Cs.copy()
        # tmpCs.insert(2, defaultC)
        # tmpStresses = maxStresses.copy()
        # tmpStresses.insert(2, defaultMaxStress)
        # tmpElementSizes = elementSizes.copy()
        # tmpElementSizes.insert(2, defaultElementSize)
        # diffCs = []
        # diffMaxStresses = []
        # for i in range(len(tmpElementSizes)-1):
        #     diffElementSizes = tmpElementSizes[i]-tmpElementSizes[i+1]
        #     diffCs.append((tmpCs[i]-tmpCs[i+1])/diffElementSizes)
        #     diffMaxStresses.append((tmpStresses[i]-tmpStresses[i+1])/diffElementSizes)


        # # Use the upper and lower bounds on element size to set the
        # # absolute tolerance and the 2x middle cases to define the
        # # relative tolerance
        # atolC = Cs[0] - Cs[-1]
        # atolStress = diffMaxStresses[0] - diffMaxStresses[-1]
        
        # self.assertAlmostEqual(defaultMaxStress, )

if __name__ == "__main__":
    unittest.main()
