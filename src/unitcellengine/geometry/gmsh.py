from pathlib import Path
import numpy as np
from unitcellengine.geometry import Geometry, DEFAULT_THICKNESS
import logging
from unitcellengine.utilities import timing, suppressStream
import sys

# Temporarily remove the current directly from the search path tl load
# the gmsh package.
tmp = sys.path.pop(0)
import gmsh as _gmsh
sys.path.insert(0, tmp)

# Create logger
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _simpleCubic(L, W, H):
    """ Simple cubic graph lines """
    L2, W2, H2 = L/2, W/2, H/2
    return [[[-L2, -W2, -H2], [-L2, -W2, H2]],
            [[-L2, W2, -H2], [-L2, W2, H2]],
            [[L2, -W2, -H2], [L2, -W2, H2]],
            [[L2, W2, -H2], [L2, W2, H2]],

            [[-L2, -W2, -H2], [-L2, W2, -H2]],
            [[L2, -W2, -H2], [L2, W2, -H2]],
            [[-L2, -W2, H2], [-L2, W2, H2]],
            [[L2, -W2, H2], [L2, W2, H2]],
            
            [[-L2, -W2, -H2], [L2, -W2, -H2]],
            [[-L2, -W2, H2], [L2, -W2, H2]],
            [[-L2, W2, -H2], [L2, W2, -H2]],
            [[-L2, W2, H2], [L2, W2, H2]],
            ]

def _bcc(L, W, H):
    """ Body centered cubic graph lines """
    L2, W2, H2 = L/2, W/2, H/2
    return [[[-L2, -W2, -H2], [L2, W2, H2]],
            [[L2, -W2, -H2], [-L2, W2, H2]],
            [[-L2, W2, -H2], [L2, -W2, H2]],
            [[L2, W2, -H2], [-L2, -W2, H2]]
            ]

def _fcc(L, W, H):
    """ Face centered cubic graph lines """
    L2, W2, H2 = L/2, W/2, H/2
    return [[[-L2, -W2, -H2], [-L2, W2, H2]],
            [[-L2, -W2, H2], [-L2, W2, -H2]],
            [[L2, -W2, -H2], [L2, W2, H2]],
            [[L2, -W2, H2], [L2, W2, -H2]],

            [[-L2, -W2, -H2], [L2, -W2, H2]],
            [[-L2, -W2, H2], [L2, -W2, -H2]],
            [[-L2, W2, -H2], [L2, W2, H2]],
            [[-L2, W2, H2], [L2, W2, -H2]],

            [[-L2, -W2, -H2], [L2, W2, -H2]],
            [[-L2, W2, -H2], [L2, -W2, -H2]],
            [[-L2, -W2, H2], [L2, W2, H2]],
            [[-L2, W2, H2], [L2, -W2, H2]],
            ]

def _column(L, W, H):
    """ Single central column """
    return [[[0, 0, -H/2], [0, 0, H/2]]]

# @TODO Columns doesn't currently work
def _columns(L, W, H):
    """ Single central column and corner columns"""
    L2, W2, H2 = L/2, W/2, H/2
    return [[[0, 0, -H2], [0, 0, H2]],
            [[-L2, -W2, -H2], [-L2, -W2, H2]],
            [[-L2, W2, -H2], [-L2, W2, H2]],
            [[L2, -W2, -H2], [L2, -W2, H2]],
            [[L2, W2, -H2], [L2, W2, H2]],
            ]

# Due to the nodal points at the corners of the unit cell, you will
# miss key volume data unless you over build the unit cell. We
# therefore output the nominal geometry, along with overhanging
# ligaments 
def _diamond(L, W, H):
    """ Diamond unit cell """
    L2, W2, H2 = L/2, W/2, H/2
    L4, W4, H4 = L/4, W/4, H/4

    lines =  [[[-L4, -W4, H4], [-L2, -W2, H2]],
            [[L4, W4, H4], [L2, W2, H2]],
            [[-L4, -W4, H4], [0, 0, H2]],
            [[L4, W4, H4], [0, 0, H2]],

            [[-L4, W4, -H4], [-L2, W2, -H2]],
            [[L4, -W4, -H4], [L2, -W2, -H2]],
            [[-L4, W4, -H4], [0, 0, -H2]],
            [[L4, -W4, -H4], [0, 0, -H2]],

            [[-L4, -W4, H4], [-L2, 0, 0]],
            [[L4, W4, H4], [L2, 0, 0]],
            [[-L4, -W4, H4], [0, -W2, 0]],
            [[L4, W4, H4], [0, W2, 0]],

            [[-L4, W4, -H4], [-L2, 0, 0]],
            [[L4, -W4, -H4], [L2, 0, 0]],
            [[-L4, W4, -H4], [0, W2, 0]],
            [[L4, -W4, -H4], [0, -W2, 0]],

            [[L2, 0, 0], [L2+L4, -W4, H4]],
            [[L2, 0, 0], [L2+L4, W4, -H4]],
            [[-L2, 0, 0], [-L2-L4, W4, H4]],
            [[-L2, 0, 0], [-L2-L4, -W4, -H4]],

            [[0, W2, 0], [L4, W2+W4, -H4]],
            [[0, W2, 0], [-L4, W2+W4, H4]],
            [[0, -W2, 0], [L4, -W2-W4, H4]],
            [[0, -W2, 0], [-L4, -W2-W4, -H4]],

            [[0, 0, H2], [L4, -W4, H2+H4]],
            [[0, 0, H2], [-L4, W4, H2+H4]],
            [[0, 0, -H2], [L4, W4, -(H2+H4)]],
            [[0, 0, -H2], [-L4, -W4, -(H2+H4)]],

            [[L2, -W2, H2], [L2-L4, -W2-W4, H2-H4]],
            [[L2, -W2, H2], [L2+L4, -W2+W4, H2-H4]],
            [[L2, -W2, H2], [L2-L4, -W2+W4, H2+H4]],

            [[-L2, W2, H2], [-L2-L4, W2-W4, H2-H4]],
            [[-L2, W2, H2], [-L2+L4, W2+W4, H2-H4]],
            [[-L2, W2, H2], [-L2+L4, W2-W4, H2+H4]],

            [[L2, W2, -H2], [L2-L4, W2-W4, -H2-H4]],
            [[L2, W2, -H2], [L2-L4, W2+W4, -H2+H4]],
            [[L2, W2, -H2], [L2+L4, W2-W4, -H2+H4]],

            [[-L2, -W2, -H2], [-L2+L4, -W2-W4, -H2+H4]],
            [[-L2, -W2, -H2], [-L2-L4, -W2+W4, -H2+H4]],
            [[-L2, -W2, -H2], [-L2+L4, -W2+W4, -H2-H4]],
           ]
    
    return lines


_GRAPH_GEOMETRY = {'Simple cubic': _simpleCubic,
                   'Body centered cubic': _bcc,
                   'Face centered cubic': _fcc,
                   'Column': _column,
                   'Columns': _columns,
                   'Diamond': _diamond}

PI2 = np.pi*2

def _schwarz(x, y, z, L, W, H, c):
    return np.cos(PI2/L*x) + np.cos(PI2/W*y) + np.cos(PI2/H*z) - c

_WALLED_TPMS_GEOMETRY = {'Schwarz': _schwarz} 

class GmshGeometry(Geometry):
    """ Unitcell geometry generation with gmsh """

    # List of graph-based unit cells in nTop. Note that below order if
    # import and must match the order shown in nTop.
    _GRAPH_UNITCELLS = ('Simple cubic',
                        'Body centered cubic',
                        'Face centered cubic',
                        'Column',
                        'Columns',
                        'Diamond',
                        'Fluorite',
                        'Octet',
                        'Truncated cube',
                        'Truncated octahedron',
                        'Kelvin cell',
                        'IsoTruss',
                        'Re-entrant',
                        'Weaire-Phelan',
                        'Triangular honeycomb',
                        'Triangular honeycomb rotated',
                        'Hexagonal honeycomb',
                        'Re-entrant honeycomb',
                        'Square honeycomb rotated',
                        'Square honeycomb',
                        'Face centered cubic foam',
                        'Body centered cubic foam',
                        'Simple cubic foam',
                        'Hex prism diamond',
                        'Hex prism edge',
                        'Hex prism vertex centroid',
                        'Hex prism central axis edge',
                        'Hex prism laves phase',
                        'Tet oct vertex centroid',
                        'Oct vertex centroid')
    
    # List of Walled TMPS-based unit cells in nTop. Note that below order is
    # import and must match the order shown in nTop.
    _WALLED_TPMS_UNITCELLS = ('Gyroid',
                            'Schwarz',
                            'Diamond',
                            'Lidinoid',
                            'SplitP',
                            'Neovius')

    _geometryExtension = "step"

    def __init__(self, unitcell, length, width, height, 
                 thickness=DEFAULT_THICKNESS, 
                 directory=".", form=None):
        """ Initialize unit cell object 
        
        Arguments
        ---------
        unitcell: str
            Specifies the unit cell name.
        length: float > 0
            Defines the normalized length of the unit cell.
        width: float > 0
            Defines the normalized width of the unit cell.
        height: float > 0
            Defines the normalized height of the unit cell.
        
        Keywords
        --------
        thickness: float > 0 (default = depends on unitcell)
            Defines the normalized thickness of the unitcell
            ligaments/walls.
        directory: str of Path (default="Database")
            Defines the base database output folder where results will
            be stored.
        form: None, "graph", or "walled tmps" (Default=None)
            Defines the unitcell form. If None, the form is
            automatically determined based on the *unitcell* name.
        
        """

        # Run the superclass constructor
        super().__init__(unitcell, length, width, height, 
                         thickness=thickness, 
                         directory=directory, form=form)

        # Define the absolute dimension that most normalization
        # reference
        self._reference = 10
    
    @property
    def mshFilename(self):
        """ Filename of the generated msh file"""
        return self.directory / Path(f"unitcellGeometry.geo")

    @timing(logger)
    def run(self, reuse=True, blocking=True):
        """ Run nTop file to generate lattice geometry 
        
        Keywords
        --------
        headless: boolean (default=True)
            Specifies whether or not to run nTop in headless mode. If
            True, nTop runs in the background while, if False, nTop
            opens a session window.
        reuse: boolean (default=True)
            Specifies whether or not to rerun the geometry generation.
            If True, existing geometry will be used if it already
            exists, otherwise, the geometry will be recreated.
        blocking: boolean (default=True)
            Specifies whether or not nTop should block further execution
            (True) or should run in the background (False). If run in
            the background, the process status can be monitored by the
            returned Popen object.
        
        Returns
        -------
        Popen object containing the nTop process
        """
        
        logger.info(f"Running geometry generation of {self}.")

        _gmsh.initialize()

        # if logger.level == logger.DEBUG:
        _gmsh.option.setNumber("General.Terminal", 1)


        # Define geometry sizing
        L = self.length*self._reference
        W = self.width*self._reference
        H = self.height*self._reference
        t = self.thickness*self._reference
        L2, W2, H2 = L/2, W/2, H/2

        boundaries = dict(xmin=-L2, xmax=L2,
                        ymin=-W2, ymax=W2,
                        zmin=-H2, zmax=H2)
        dims = dict(x=L, y=W, z=H)

        # lines = [[[-L2, -W2, -H2], [L2, W2, H2]],
        #         [[L2, -W2, -H2], [-L2, W2, H2]],
        #         [[-L2, W2, -H2], [L2, -W2, H2]],
        #         [[L2, W2, -H2], [-L2, -W2, H2]]
        #         ]
        
        lines = _GRAPH_GEOMETRY[self.unitcell](L, W, H)

        # Create geometric bodies
        _gmsh.model.occ.addBox(-L2, -W2, -H2, L, W, H, 10)
        base = 20
        cylinders = []
        for i, ((x1, y1, z1), (x2, y2, z2)) in enumerate(lines):
            
            # Parse the coordinates into relevant GMSH paremeters
            dx, dy, dz = x2-x1, y2-y1, z2-z1

            # Do a quick spot check that the ligament has positive
            assert dx**2 + dy**2 + dz**2 > 0

            # Create the ligament
            cylinder = _gmsh.model.occ.addCylinder(x1, y1, z1, 
                                                   dx, dy, dz, t/2)
            cylinders.append((3, cylinder))
            
            # Increment the ligament counter
            i += 1

        # _gmsh.model.occ.synchronize()
        # _gmsh.fltk.run()
        
        # if i > 0:
        #     # Boolean the geometries together
        #     # [(3, base+tag+1) for tag in range(len(lines)-1)]
        #     # fuse = _gmsh.model.occ.fuse([(3, base+tag+1) for tag in range(len(lines)-1)],
        #     #                             [(3, base)])
        fuse = _gmsh.model.occ.fuse(cylinders, cylinders)
        #     # ref = fuse[0] + [x[0] for x in fuse[1] if x]
        #     ref = fuse[0]
        # else:
        #     ref = [(3, base)]
        
        # Cut of the edges to create the final unit cell body
        intersect = _gmsh.model.occ.intersect(fuse[0], [(3, 10)])

        # Synchronize all of the model updates
        _gmsh.model.occ.synchronize()

        # Ask OpenCASCADE to compute more accurate bounding boxes of
        # entities 
        _gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)

        # Calculate the relative density of the structure
        volume = sum([_gmsh.model.occ.getMass(*body) for body in intersect[0]])
        relativeDensity = volume/(L*W*H)
        logger.info(f"Geometry relative density: {relativeDensity*100:.2f}%")

            
        # Export the geometry
        _gmsh.write(self.geometryFilename.as_posix())


        # Export unit cell properties
        
        # Close gmsh interface
        _gmsh.finalize()

if __name__ == "__main__":
    design = GmshGeometry('Diamond', 4, 1.25, 1, thickness=0.3, form='graph',
                          directory=Path(__file__).parent/Path("tests"))
    # design = GmshGeometry('Diamond', 1.5, 5, 1.25, T=0.3, form='graph',
    #                       directory=Path(r"F:\Lattice\Database"))
    design._geometryExtension = "stl"
    p = design.run()
    print(design)
