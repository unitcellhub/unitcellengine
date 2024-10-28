# UnitcellEngine

> [!WARNING]
> This is alpha software that is in active development and is likely to experience breaking changes.

*UnitcellEngine* is part of the *UnitcellHub* software ecosystem, providing functionality to generate and simulate lattice unitcell geometries.
Written fully in Python, it brings together state-of-the-art technology like implicit geometry representation, meshless finite element analysis, and homogenization theory, to provide an automated yet robust pipeline.

![](docs/source/images/overview.png)
 
## Installation

UnitcellEngine only supports Python 3. 
If running on a Windows machine, you may sometimes need to first install Microsoft Visual C++ for some of the dependencies to install correctly (such as number and scikit-learn). See https://visualstudio.microsoft.com/visual-cpp-build-tools/ for details on how to install the Windows C++ Build Tools; 
additionally, due to performance limitations of the openblas-based installation of numpy via pip, it is recommended that conda be used to setup the base numpy/scipy environment.

For the current stable release

```
pip install unitcellengine

```

## Examples

*UnitcellEngine* has a range of built-in lattice geometry forms, including graph-style (such as beam and plate) and thin walled Triply Periodic Minimal Surface (TPMS) lattices.
The general workflow is to define the unitcell, generate the geometry, generate the mesh, and solve the relevant homogenization problems.
Each step of the process calculates relevant quantities and stores them in output files that feed into subsequent calculations (for example, a mesh file is required before a homogenization can be run).

```python
from unitcellengine.design import UnitcellDesign, UnitcellDesigns
import numpy as np

# Set numpy precision to allow for better printout of homogenized matrices
np.set_printoptions(precision=4)

# Define unitcell
design = UnitcellDesign('Octet', 1, 1, 1, thickness=0.1)

# The default behavior is to reuse existing date. Set to False to force regeneration.
reuse = True

# Generate geometry (which calculated propertes like relative denstiy and relative surface area)
design.generateGeometry(reuse=reuse)

# Generate mesh
design.generateMesh(reuse=reuse)

# Calculate homogenized elastic properties
design.homogenizationElastic.run(reuse)

# Post process homogenization results
design.homogenizationElastic.process()

# Calculate homogenized conductance properties
design.homogenizationConductance.run(reuse)

# Post process conductance results
design.homogenizationConductance.process()

# Print the homogenizated stiffness matrix
print(design.homogenizationElastic)
print(design.homogenizationConductance)
```
The results are by default stored in the folder structure "Database/<unitcell form>/<unitcell type>/<LX_XXX_WY_YYY_HZ_ZZZ_TU_UUU>", where <unitcell form> is either "graph" or "walledtpms", "unicell type" is the unitcell type, and "<LX_XXX_WY_YYY_HZ_ZZZ_TU_UUU>" is a folder with the Length, Width, Height, and Thickness properties of the lattice.
For example, in the case above, it will store the results in "Database/graph/Octet/L1_000_W1_000_H1_000_T0_100".

For more usage examples, see [examples/examples.ipynb](examples/examples.ipynb).

## High Performance Computing and remote Linux servers
When running on remote systems, there are a few features that don't work straight out of the box. 
If geometry rendering is required (which relies on VTK), additional utitilies will likely need to be installed.

### CENTOS and RH systems without root privileges
Xvfb is an X server that can run on machines with no display hardware and no physical input devices. 
It emulates a dumb frame buffer using virtual memory. 
Install xvfb following the steps defined here: https://stackoverflow.com/questions/36651091/how-to-install-packages-in-linux-centos-without-root-user-with-automatic-depen. 
Note, on most HPC systems, these files should be installed on a high performance file system rather than a network drive. 

In addition to xvfb, you will need to install the OpenGL libraries "mesa-libGL" and "mesa-libGL-devel" using the same steps as for xvfb. 
In some cases, you might still get an "GLSL 1.50 is not supported. 
Supported versions are: 1.10, 1.20, 1.30, 1.00 ES, and 3.00 ES" error. 
In this case, add the following environmental variable

```
export MESA_GL_VERSION_OVERRIDE=3.2
```
