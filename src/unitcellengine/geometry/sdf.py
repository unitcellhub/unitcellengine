import sys

tmp = sys.path.pop(0)
from sdf.d3 import capped_cylinder, box, rounded_box, op3
import sdf.d3 as d3
from sdf.d2 import polygon, rectangle
import sdf
sys.path.insert(0, tmp)
from pathlib import Path
import numpy as np
from unitcellengine.geometry import Geometry, DEFAULT_ELEMENT_SIZE, \
                              DEFAULT_THICKNESS, cachedProperty
import logging
from unitcellengine.utilities import timing, Timing, suppressStream
from numpy import sin, cos, pi
import trimesh
import json
import pyvista as pv
import json
import re
from skimage import measure

# Set the default PyVista setup to have a transparent background
pv.global_theme.transparent_background = True

# Create logger
logger = logging.getLogger(__name__)

PI2 = pi*2

# Import graph geometry definitions
with open(Path(__file__).parent / Path("definitions/definitions.json"), 'r') as f:
    GRAPH_DEF = json.load(f)

# Define graph geometry repetition padding. Setting the repetition
# padding to a value of 1, looks out side the unitcell for geometry
# information, which is important for graph unit cells that have
# diagonal elements that meet at the boundaries of the unit cell (in
# particular, capturing the correct fillet details). Unfortunately, this
# padding behavior can cause some undesired features, such as in the
# re-entrant unitcell, where some of the features connect Across the
# unit cell that shouldn't. It can also cause unwanted "bulging" due to
# the union operation. For this reason, padding is turned off unless it
# is explicitly needed.
# @NOTE: This addition of padding has a pretty significant increase in
# computation time.
_GRAPH_PADDING = {unitcell: 0 for unitcell in GRAPH_DEF.keys()}
for unitcell in ['Body centered cubic', 'Diamond', 'Flourite', 
                 'Truncated octahedron', 'Kelvin cell']:
    _GRAPH_PADDING[unitcell] = 1

# @NOTE: The addition of padding to the triangular honeycombs fixes filleting in
# adjoining connections, but does cause an unwanted bulge in the perpedicular
# ligaments on these faces. Because the fillet is likely more useful than the
# bulge is detrimental, we'll leave it in for now.
_GRAPH_PADDING['Triangular honeycomb rotated'] = [0, 1, 0]
_GRAPH_PADDING['Triangular honeycomb'] = [1, 0, 0]

#@todo: Need to tune thicknesses to match the ntopology definitions
# https://support.ntopology.com/hc/en-us/articles/360053267814-What-equations-are-used-to-create-the-TPMS-types-
# https://support.ntopology.com/hc/en-us/articles/4403371438611
# https://support.ntopology.com/hc/en-us/articles/360061475314-Why-isn-t-the-TPMS-Lattice-thickness-constant-when-I-distort-it-


# def _worker(sdf, points):
#     """ Parallel processing worker for SDF evaluation """
#     return sdf.f(points)

@sdf.sdf3
def _gyroid(L, W, H):
    def f(p):
        X = PI2/L*p[:, 0]
        Y = PI2/W*p[:, 1]
        Z = PI2/H*p[:, 2]

        # Define the nominal implicit function
        phi0 = (sin(X)*cos(Y) + sin(Y)*cos(Z) + sin(Z)*cos(X))

        # Calculate the gradient of the implicit function to convert to
        # sign distance function
        dphidx = PI2/L*(cos(X)*cos(Y)-sin(Z)*sin(X))
        dphidy = PI2/W*(cos(Y)*cos(Z)-sin(X)*sin(Y))
        dphidz = PI2/H*(cos(Z)*cos(X)-sin(Y)*sin(Z))
        norm = np.sqrt(dphidx**2+dphidy**2+dphidz**2)

        # Because the gradient of the implicit function is zero on the 
        # surface, the gradient is zero and hence hare a zero norm. So,
        # replace all NaN with 0/
        out = phi0/norm
        out[np.isnan(out)] = 0

        return out

    return f

@sdf.sdf3
def _schwarz(L, W, H):
    def f(p):
        X = PI2/L*p[:, 0]
        Y = PI2/W*p[:, 1]
        Z = PI2/H*p[:, 2]

        # Define the nominal implicit function
        phi0 = -(cos(X) + cos(Y) + cos(Z))

        # Calculate the gradient of the implicit function to convert to
        # sign distance function
        dphidx = PI2/L*sin(X)
        dphidy = PI2/W*sin(Y)
        dphidz = PI2/H*sin(Z)
        norm = np.sqrt(dphidx**2+dphidy**2+dphidz**2)

        # Because the gradient of the implicit function is zero on the 
        # surface, the gradient is zero and hence hare a zero norm. So,
        # replace all NaN with 0
        out = phi0/norm
        out[np.isnan(out)] = 0

        return out

    return f

@sdf.sdf3
def _iwp(L, W, H):
    def f(p):
        X = PI2/L*p[:, 0]
        Y = PI2/W*p[:, 1]
        Z = PI2/H*p[:, 2]

        # Define the nominal implicit function
        phi = 2*(cos(X)*cos(Y) + cos(Y)*cos(Z) + cos(Z)*cos(X)) -\
                (cos(2*X)+cos(2*Y)+cos(2*Z))

        # Calculate the gradient of the implicit function to convert to
        # sign distance function
        dphidx = PI2/L*(-2*sin(X)*cos(Y) - 2*sin(X)*cos(Z) + 2*sin(2*X))
        dphidy = PI2/W*(-2*sin(Y)*cos(X) - 2*sin(Y)*cos(Z) + 2*sin(2*Y))
        dphidz = PI2/H*(-2*sin(Z)*cos(X) - 2*sin(Z)*cos(Y) + 2*sin(2*Z))
        norm = np.sqrt(dphidx**2+dphidy**2+dphidz**2)

        # Because the gradient of the implicit function is zero on the 
        # surface, the gradient is zero and hence hare a zero norm. So,
        # replace all NaN with 0/
        out = phi/norm
        out[np.isnan(out)] = 0

        return out

    return f

@sdf.sdf3
def _diamond(L, W, H):
    def f(p):
        X = PI2/L*p[:, 0]
        Y = PI2/W*p[:, 1]
        Z = PI2/H*p[:, 2]

        # Define the nominal implicit function
        phi =   sin(X)*sin(Y)*sin(Z) + \
                sin(X)*cos(Y)*cos(Z) + \
                cos(X)*sin(Y)*cos(Z) + \
                cos(X)*cos(Y)*sin(Z)
    
    
        # Calculate the gradient of the implicit function to convert to
        # sign distance function
        dphidx = PI2/L*(-sin(X)*sin(Y)*cos(Z) - sin(X)*sin(Z)*cos(Y) + \
                        sin(Y)*sin(Z)*cos(X) + cos(X)*cos(Y)*cos(Z))
        dphidy = PI2/W*(-sin(X)*sin(Y)*cos(Z) + sin(X)*sin(Z)*cos(Y) - \
                        sin(Y)*sin(Z)*cos(X) + cos(X)*cos(Y)*cos(Z))
        dphidz = PI2/H*(sin(X)*sin(Y)*cos(Z) - sin(X)*sin(Z)*cos(Y) - \
                        sin(Y)*sin(Z)*cos(X) + cos(X)*cos(Y)*cos(Z))
        norm = np.sqrt(dphidx**2+dphidy**2+dphidz**2)

        # Because the gradient of the implicit function is zero on the 
        # surface, the gradient is zero and hence hare a zero norm. So,
        # replace all NaN with 0/
        out = phi/norm
        out[np.isnan(out)] = 0

        return out

    return f

@sdf.sdf3
def _lidinoid(L, W, H):
    def f(p):
        X = PI2/L*p[:, 0]
        Y = PI2/W*p[:, 1]
        Z = PI2/H*p[:, 2]

        # Define the nominal implicit function
        phi = (sin(2*X)*cos(Y)*sin(Z) + \
                sin(2*Y)*cos(Z)*sin(X) + \
                sin(2*Z)*cos(X)*sin(Y) - \
                cos(2*X)*cos(2*Y) - \
                cos(2*Y)*cos(2*Z) - \
                cos(2*Z)*cos(2*X)+0.3)
        
        # return phi
        # Calculate the gradient of the implicit function to convert to
        # sign distance function
        dphidx = PI2/L*(-sin(X)*sin(Y)*sin(2*Z) + 2*sin(2*X)*cos(2*Y) +\
                        2*sin(2*X)*cos(2*Z) + sin(2*Y)*cos(X)*cos(Z) +\
                        2*sin(Z)*cos(2*X)*cos(Y))
        dphidy = PI2/W*(2*sin(X)*cos(2*Y)*cos(Z) - \
                        sin(2*X)*sin(Y)*sin(Z) + 2*sin(2*Y)*cos(2*X) + \
                        2*sin(2*Y)*cos(2*Z) + sin(2*Z)*cos(X)*cos(Y))
        dphidz = PI2/H*(-sin(X)*sin(2*Y)*sin(Z) + \
                        sin(2*X)*cos(Y)*cos(Z) + \
                        2*sin(Y)*cos(X)*cos(2*Z) + 2*sin(2*Z)*cos(2*X) +\
                        2*sin(2*Z)*cos(2*Y))
        norm = np.sqrt(dphidx**2+dphidy**2+dphidz**2)

        # Because the gradient of the implicit function is zero on the 
        # surface, the gradient is zero and hence hare a zero norm. So,
        # replace all NaN with 0/
        out = phi/norm
        out[np.isnan(out)] = 0

        return out

    return f

@sdf.sdf3
def _splitp(L, W, H):
    def f(p):
        X = PI2/L*p[:, 0]
        Y = PI2/W*p[:, 1]
        Z = PI2/H*p[:, 2]
        phi = (1.1*(sin(2*X)*sin(Z)*cos(Y) + \
                     sin(2*Y)*sin(X)*cos(Z) + \
                     sin(2*Z)*sin(Y)*cos(X)) - \
                0.2*(cos(2*X)*cos(2*Y) + \
                     cos(2*Y)*cos(2*Z) + \
                     cos(2*Z)*cos(2*X)) - \
                0.4*(cos(2*X)+cos(2*Y)+cos(2*Z)))
        
        # Calculate the gradient of the implicit function to convert to
        # sign distance function
        dphidx = PI2/L*(-1.1*sin(X)*sin(Y)*sin(2*Z) + \
                        0.4*sin(2*X)*cos(2*Y) + 0.4*sin(2*X)*cos(2*Z) +\
                        0.8*sin(2*X) + 1.1*sin(2*Y)*cos(X)*cos(Z) + \
                        2.2*sin(Z)*cos(2*X)*cos(Y))
        dphidy = PI2/W*(2.2*sin(X)*cos(2*Y)*cos(Z) - \
                        1.1*sin(2*X)*sin(Y)*sin(Z) + \
                        0.4*sin(2*Y)*cos(2*X) + 0.4*sin(2*Y)*cos(2*Z) + \
                        0.8*sin(2*Y) + 1.1*sin(2*Z)*cos(X)*cos(Y))
        dphidz = PI2/H*(-1.1*sin(X)*sin(2*Y)*sin(Z) + \
                        1.1*sin(2*X)*cos(Y)*cos(Z) + \
                        2.2*sin(Y)*cos(X)*cos(2*Z) + \
                        0.4*sin(2*Z)*cos(2*X) + 0.4*sin(2*Z)*cos(2*Y) + \
                        0.8*sin(2*Z))
        norm = np.sqrt(dphidx**2+dphidy**2+dphidz**2)

        # Because the gradient of the implicit function is zero on the 
        # surface, the gradient is zero and hence hare a zero norm. So,
        # replace all NaN with 0/
        out = phi/norm
        out[np.isnan(out)] = 0

        return out

    return f

@sdf.sdf3
def _neovius(L, W, H):
    def f(p):
        X = PI2/L*p[:, 0]
        Y = PI2/W*p[:, 1]
        Z = PI2/H*p[:, 2]
        phi = (3*(cos(X)+cos(Y)+cos(Z))+4*cos(X)*cos(Y)*cos(Z))

        # Calculate the gradient of the implicit function to convert to
        # sign distance function
        dphidx = PI2/L*(-4*sin(X)*cos(Y)*cos(Z) - 3*sin(X))
        dphidy = PI2/W*(-4*sin(Y)*cos(X)*cos(Z) - 3*sin(Y))
        dphidz = PI2/H*(-4*sin(Z)*cos(X)*cos(Y) - 3*sin(Z))
        norm = np.sqrt(dphidx**2+dphidy**2+dphidz**2)

        # Because the gradient of the implicit function is zero on the 
        # surface, the gradient is zero and hence hare a zero norm. So,
        # replace all NaN with 0/
        out = phi/norm
        out[np.isnan(out)] = 0

        return out

    return f

@op3
def rotateM(other, matrix):
    def f(p):
        return other(np.dot(p, matrix))
    return f

_min = np.minimum
_max = np.maximum
_abs = np.abs
_nax = np.newaxis
_dot = np.dot
@sdf.sdf2
def triangle(points):
    assert len(points) == 3, f"Triangle is defined by 3 points, not {len(points)}."
    points = [np.array(p) for p in points]
    # @njit
    def f(p):
        # From https://www.shadertoy.com/view/XsXSz4
        # Create triangle line segments
        es = [points[i]-points[i-1] for i in [1, 2, 0]]

        # For each point, create a vector to each line segment
        vs = [p - p0 for p0 in points]

        # s = es[0][0]*es[2][1] - es[0][1]*es[2][0]
        # mins = np.ones((2, p.shape[0]))
        # for i in range(3):
        #     e = es[i]
        #     v = vs[i]
        #     clip = _dot(v, e)/_dot(e, e)
        #     clip[clip > 1] = 1
        #     clip[clip < 0] = 0
        #     pq = v - e[None, :]*clip[:, None]
        #     pair = np.vstack((np.sum(pq*pq, axis=1), 
        #                        s*(v[:, 0]*e[1]-v[:, 1]*e[0])))
        #     if i == 1:
        #         mins[:] = pair[:]
        #     else:
        #         mins = np.minimum(mins, pair)
        # Calculate perpendicular vectors
        pqs = [v - e[_nax]*np.clip(_dot(v, e)[_nax].T/_dot(e, e), 0., 1.) 
                                                for e, v in zip(es, vs)]

        # Calculate the squared distance for each point and the
        # corresponding sign (is it inside or outside?)
        s = es[0][0]*es[2][1] - es[0][1]*es[2][0]
        pairs = [np.vstack((np.linalg.norm(pq, axis=1), 
                             s*(v[:, 0]*e[1]-v[:, 1]*e[0])))
                                    for e, v, pq in zip(es, vs, pqs)]
        # Calculate the minimum distance
        mins = np.min(pairs, axis=0)

        return mins[0, :]*np.sign(mins[1, :])
    return f

def _plate(points, t):
    """ Create a plate sdf representation based on defined points """
    N = len(points)
    
    # Calculate the centroid of the points
    centroid = sum(points)/N
    
    # Calculate the plate coordinate system
    # Note, this assumes all of the points are coplanar
    xp = points[1] - points[0]
    xp /= np.linalg.norm(xp)
    v2 = points[2] - points[1]
    v2 /= np.linalg.norm(v2)
    zp = np.cross(xp, v2)
    zp /= np.linalg.norm(zp)
    yp = np.cross(zp, xp)
    yp /= np.linalg.norm(yp)

    # Create the rotation matrix from the nominal geometry space to the 
    # desired space
    # https://stackoverflow.com/questions/29754538/rotate-object-from-one-coordinate-system-to-another
    M = np.vstack((xp, yp, zp)).T

    if N == 3:
        # Create triangular plate

        # Shift the triangle to the origin and rotate into 2D plane
        ps = np.dot(M.T, np.array(points).T-centroid.reshape(-1, 1))

        # Create 2D object
        # tria = polygon([p[:2] for p in ps.T])
        tria = triangle([p[:2] for p in ps.T])

        # Extrude, rotate and translate 2D planar object
        return tria.extrude(t).rotateM(M).translate(centroid)

    elif N == 4:
        # Create rectangular plate
        # Calculate length and width
        L = np.linalg.norm(points[1]-points[0])
        W = np.linalg.norm(points[2]-points[1])

        # Create plate in nonoriented condition and then orient it accordingly
        # Note, we're using the rounded_box here to cap the plates, which
        # results in better repetition and unioning performance
        return  rounded_box([L+t, W+t, t], t/2).rotateM(M).translate(centroid)
    else:
        raise RuntimeError(f"Incorrect number of points {N}.",
                            "Only 3 or 4 points are valid.")

@op3
def unionMod(a, *bs, k=None, eps=1e-6):
    def f(p):
        d1 = a(p)
        n1 = np.hstack((a(p+np.array([eps, 0, 0])[_nax])-d1,
                        a(p+np.array([0, eps, 0])[_nax])-d1,
                        a(p+np.array([0, 0, eps])[_nax])-d1))
        n1 /= np.linalg.norm(n1, axis=1)[_nax].T
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            if K is None:
                d1 = _min(d1, d2)
            else:
                n2 = np.hstack((b(p+np.array([eps, 0, 0])[_nax])-d2,
                                b(p+np.array([0, eps, 0])[_nax])-d2,
                                b(p+np.array([0, 0, eps])[_nax])-d2))
                n2 /= np.linalg.norm(n2, axis=1)[_nax].T

                align = (1 - ((n1*n2).sum(axis=1)))[_nax].T

                h = align*_max(K-_abs(d1-d2), 0.0)/K
                d1 = _min(d1, d2) - h*h*K*(1.0/4.0)
                n1 = (n1+n2)
                n1 /= np.linalg.norm(n1, axis=1)[_nax].T
        return d1
    return f

# @op3
# def smooth(other, r=0.):
#     def f(p):

#         # # Calculate centerpoints
#         # d = other(p)

#         # # Define smoothing ball
#         # Theta, Phi = np.meshgrid(np.linspace(0, np.pi, 4), 
#         #                          np.linspace(0, 2*np.pi, 4))
#         # ball = [np.array([np.cos(phi)*np.cos(theta),
#         #                   np.sin(phi)*np.sin(theta),
#         #                   np.cos(theta)]) for phi, theta in zip(Phi.flatten(), 
#         #                                                         Theta.flatten())]
#         # ball = r*np.array(ball)

#         # pr = other((d[_nax]+ball[_nax].transpose((1, 0, 2))).reshape((-1, 3), order="F"))
#         # d += pr.reshape((ball.shape[0], -1)).sum(axis=0)[_nax].T
#         # d /= ball.shape[0]+1
#         # return d
#     return f

def _graph(unitcell):
    """ Graph based unitcell geometry"""
    def f(L, W, H, t, smoothing=0):
        # Define unitcell scaling
        scaling = np.array([L, W, H])
        
        # Pull out unitcell definition
        definition = GRAPH_DEF[unitcell]
        nodes = definition['node']
        beams = definition['beam']
        faces = definition['face']
        geom1 = None
        geom2s = []

        # Create beam-based objects
        r = t/2
        for v in beams.values():
            z1, z2 = [np.array([float(nodes[str(subv)][d]) for d in 'xyz'])*scaling 
                        for subv in v.values()] 
            
            tmp = sdf.capsule(z1, z2, r).repeat([L, W, H], 
                                                 padding=_GRAPH_PADDING[unitcell])
            if not geom1:
                geom1 = tmp
            else:
                geom2s.append(tmp)
        
        # Create plate-based objects
        for v in faces.values():
            pts = [np.array([float(nodes[str(subv)][d]) for d in 'xyz'])*scaling 
                        for subv in v.values()] 
            
            tmp = _plate(pts, t).repeat([L, W, H], padding=_GRAPH_PADDING[unitcell])
            if not geom1:
                geom1 = tmp
            else:
                geom2s.append(tmp)
        
        # Combine all of the geometry
        if geom2s:
            if not (smoothing > 0):
                smoothing = None
            geom = geom1.union(*geom2s, k=smoothing)
        else:
            geom = geom1
    
        return geom
    return f

def _walledtpms(unitcell):
    """ Walled TPMS based unitcell geometry """

    def f(L, W, H, t):
        # Shell the geometry with the appropriate thickness
        # Note: this shelling process doesn't result in the exact wall
        # thickness specified as the TPMS implicit functions aren't true
        # signed distance functions. This implementation attempts to
        # create a signed distance function representation, but the
        # conversion isn't perfect.
        return unitcell(L, W, H).shell(t).repeat([L, W, H], padding=0)
    
    return f

# Assign SDF definitions to each unitcell geometry
reInvalid = re.compile("^(Hex |Tet |Oct ).*")
_SDFS_GRAPH = {unitcell: _graph(unitcell) for unitcell in GRAPH_DEF.keys()
                if not reInvalid.match(unitcell)}

_SDFS_WALLED_TPMS = {k: _walledtpms(v) for k, v in {"Gyroid": _gyroid,
                                                "Schwarz": _schwarz,
                                                "Diamond": _diamond,
                                                "Lidinoid": _lidinoid,
                                                "SplitP": _splitp,
                                                "Neovius": _neovius,
                                                "IWP": _iwp}.items()}

_GRAPH_UNITCELLS = tuple(_SDFS_GRAPH.keys())
_WALLED_TPMS_UNITCELLS = tuple(_SDFS_WALLED_TPMS.keys())

class SDFGeometry(Geometry):
    # List of graph-based unit cells in nTop. Note that below order if
    # import and must match the order shown in nTop.
    _GRAPH_UNITCELLS = _GRAPH_UNITCELLS
    
    # List of Walled TMPS-based unit cells in nTop. Note that below order is
    # import and must match the order shown in nTop.
    _WALLED_TPMS_UNITCELLS = _WALLED_TPMS_UNITCELLS

    _propertiesExtension = 'json'
    _geometryExtension = 'stl'
    

    def __init__(self, unitcell, length, width, height, 
                 thickness=DEFAULT_THICKNESS, radius=0., directory=".", 
                 elementSize=DEFAULT_ELEMENT_SIZE, form=None):
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
        radius: float >= 0 (default = 0)
            Defines the normalized smoothing radius between ligaments. 
            This is only valid for graph based ligaments. Normalization
            is with respect to the ligament/wall thickness.
        directory: str of Path (default="Database")
            Defines the base database output folder where results will
            be stored.
        elementSize: float > 0 (Default=0.2)
            Normalize size of the STL facet elements.
        form: None, "graph", or "walled tmps" (Default=None)
            Defines the unitcell form. If None, the form is
            automatically determined based on the *unitcell* name.
        
        Notes
        -----
        - The walled TPMS geometry here is similar to, but different
          than nTopology's definitions: the baseline zero thickness
          surfaces are the same, but this implementation targets a
          uniform thickening while nTopologies is non-uniform.
        - For walled TPMS geometry, the input thickness parameter isn't
          exactly replicated in the geometry, although, it should be
          within roughly 10% of the specified value. This has to do with
          the fact that the standard implicit representation of walled
          TPMS geometries is not a true sign distance function. This
          implementation attempts to convert to a sign distance
          function, but this conversion isn't perfect.
        """

        # Run the superclass constructor
        super().__init__(unitcell, length, width, height, 
                         thickness=thickness, radius=radius,
                         directory=directory, form=form)

        self.elementSize = elementSize
        self._normalization = 10

        # Create signed distance function
        L = self.DIMENSION*self.length
        W = self.DIMENSION*self.width
        H = self.DIMENSION*self.height
        T = self.DIMENSION*self.thickness
        R = T*radius

        # Define the sign distance field based on the cell type
        if self._cellForm == 'graph':
            self.sdf = _SDFS_GRAPH[self.unitcell](L, W, H, T, R)
        elif self._cellForm == 'walledtpms':
            self.sdf = _SDFS_WALLED_TPMS[self.unitcell](L, W, H, T)
            if self.unitcell.lower() in ['gyroid', 'diamond']: 
                self.sdf = self.sdf.translate([L/4, 0, 0])
        else:
            raise ValueError(f"Unexpected cell form '{self._cellForm}'. "
                             "Should be either 'graph' or 'walledtpms'.")
        
        # Make the unit cell periodic
        # self.sdf = self.sdf.repeat([L, W, H], padding=1)

    # Overload processed definition as the geometry file isn't natively
    # generated with this geometry definition
    @property
    def processed(self):
        """ Has the design been processed? """
        try:
            if self.propertiesFilename.exists():
                return True
            else:
                return False
        except:
            return False

    # @timing(logger)
    # def _evaluate(self, func, L, W, H, T, elementSize):
    #     """ Evaluate the sign distance function over a grid """

    #     WORKERS = multiprocessing.cpu_count()

    #     # Create a coarse sampling grid based on the thickness of the
    #     # geometry 
    #     sNX, sNY, sNZ = [int(np.ceil(dim/T)*2+1) for dim in [L, W, H]]
        
    #     # Create a fine scale spacing based on this coarse sampling size
    #     # and the desired resolution
    #     N = int(np.ceil(1/self.elementSize))
    #     NX, NY, NZ = [int(n*N)+1 for n in [sNX, sNY, sNZ]]

    #     logger.debug(f"Discretizing with ({NX}, {NY}, {NZ}) points "
    #                  f"({NX*NY*NZ:,} total).")
    #     X, Y, Z = np.meshgrid(np.linspace(-L/2, L/2, NX),
    #                           np.linspace(-W/2, W/2, NY),
    #                           np.linspace(-H/2, H/2, NZ))
    #     grid = pv.StructuredGrid(X, Y, Z)
    #     points = np.vstack((X.ravel(order='F'), 
    #                         Y.ravel(order='F'), 
    #                         Z.ravel(order='F'))).T
        
    #     # Create an index array for the meshed grid
    #     fullinds = np.arange(X.size).reshape((NY, NX, NZ), 
    #                                             order='F')
        
    #     # Initialize the grid point values all to 1 (which implies they
    #     # are intiailly exterior points)
    #     values = np.ones((X.size, 3))


    #     # Evaluate the sdf using a coarse sampling (on the order of 
    #     # the ligament/plate thickness) first
    #     if self.elementSize < 0.5:
    #         # Create an index array for the meshed grid
    #         fullinds = np.arange(X.size).reshape((NY, NX, NZ), 
    #                                                 order='F')
    #         values = np.ones((X.size, 3))

    #         # Check that the sampling grid is finer than the coarse
    #         # sampling 
    #         if sNX == sNY == sNZ == 1:
    #             values = func(points)
    #             logger.debug("Coarse grid is the same as the full grid."
    #                             " No further sampling required.")
    #         else:
    #             # Create a resampling grid that is slightly larger than the geometry thickness
    #             midX, midY, midZ = int(NX/2), int(NY/2), int(NZ/2)
    #             subgrid = fullinds[midY-N-2:midY+N+2, 
    #                             midX-N-2:midX+N+2, 
    #                             midZ-N-2:midZ+N+2] - \
    #                                     fullinds[midY, midX, midZ]  
    #             subgrid = subgrid.ravel(order='F')

    #             logger.debug("Coarse sampling with "
    #                         f"{X[::N,::N,::N].transpose((1, 0, 2)).shape} "
    #                         f"points ({X[::N,::N,::N].size:,} total).")
                
    #             # Calculate the SDF on the coarse grid
    #             subpoints = np.vstack((X[::N,::N,::N].ravel(order='F'), 
    #                                 Y[::N,::N,::N].ravel(order='F'), 
    #                                 Z[::N,::N,::N].ravel(order='F'))).T
    #             svalues = func(subpoints)
    #             # ax.scatter(subpoints[:, 0], subpoints[:, 1], subpoints[:, 2], marker="x")
    #             # Determine locations to subsample
    #             inds = svalues <= 0

    #             # Find neighboring points on the fine mesh
    #             logger.debug("Finding neighboring points to interior points "
    #                             "found during the coarse sampling process.")

    #             subinds = np.unique((fullinds[::N,::N,::N].ravel(order='F')[inds[:, 0]][:, None] +\
    #                                 subgrid[None, :]).ravel(order='F'))
    #             subinds = subinds[subinds > 0]
    #             subinds = subinds[subinds < fullinds.size]
    #             rinds = subinds
                
    #             # tree = spatial.KDTree(points)
    #             # resample = tree.query_ball_point(subpoints[inds[:, 0], :], 
    #             #                                 1.25*T,
    #             #                                 workers=-1)

    #             # Subsample the coarse cells that have interior points
    #             # rinds = np.unique(np.hstack(resample))
    #             logger.debug(f"Fine sampling over {rinds.size:,} points.")
    #             resamplepoints = points[rinds, :]
    #             values[rinds] = func(resamplepoints)

    @timing(logger)
    def _mesh(self, func, L, W, H, elementSize):
        """ Mesh sign distance function over given domain """

        # # Use built in generation routine (which is written in parallel)
        # points = func.generate(step=elementSize/10, 
        #                       bounds=((-L/2, -W/2, -H/2),
        #                               (L/2, W/2, H/2)),
        #                       sparse=False)
        
        # # Convert to mesh
        # verts, faces = np.unique(points, axis=0, return_inverse=True)
        # faces = faces.reshape((-1, 3))


        # Discretize space
        NX = int(np.ceil(L/elementSize))
        NY = int(np.ceil(W/elementSize))
        NZ = int(np.ceil(H/elementSize))
        x = np.linspace(-L/2, L/2, NX)
        y = np.linspace(-W/2, W/2, NY)
        z = np.linspace(-H/2, H/2, NZ)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        # Note: ij indexing is required. Otherwise, you get rotated
        # geometry
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij') 
        p = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        
        # Evaluate sign distance function on discretized space
        logger.info("Discretizing sign distance function")
        points = func.f(p)

        # Run marching cube algorithm to create surface definition
        logger.info("Generating geometric surfaces")
        vol = points.reshape(X.shape)
        verts, faces, normals, values = measure.marching_cubes(vol, 0)

        # Rescale and shift the marching cube results back to the 
        # appropriate geometric space
        scaling = np.ones(verts.shape)
        scaling[:, 0] = dx
        scaling[:, 1] = dy
        scaling[:, 2] = dz
        offset = np.zeros(verts.shape)
        offset[:, 0] = x[0]
        offset[:, 1] = y[0]
        offset[:, 2] = z[0]
        verts = verts*scaling + offset

        # Create STL mesh representation
        return trimesh.Trimesh(vertices=verts, faces=faces)

    @timing(logger)
    def visualizeVTK(self, nx=1, ny=1, nz=1):
        """ Create a StructuredGrid visualized of the geometry 
        
        Keywords
        ---------
        nx: positive integer (default=1)
            Number of cells to plot in the x direction
        ny: positive integer (default=1)
            Number of cells to plot in the y direction
        nz: positive integer (default=1)
            Number of cells to plot in the z direction
        
        """
        logger.debug(f"Creating VTK visualization of {self} with "
                     f"{nx}x{ny}x{nz} periodic cells")

        # Pull out the absolute dimensions
        L = self.DIMENSION*self.length*nx
        W = self.DIMENSION*self.width*ny
        H = self.DIMENSION*self.height*nz
        T = self.DIMENSION*self.thickness
        # elementSize = T
        elementSize = T*self.elementSize

        # Define the boxed sdf function
        func = self.sdf
        # func = self.sdf & sdf.box([L, W, H])

        # Apply margin to the geometry bounds to ensure boundary
        # features are captured correctly
        # margin = 1.05
        # L *= margin
        # W *= margin
        # H *= margin

        # Create a coarse sampling grid based on the thickness of the
        # geometry 
        sNX, sNY, sNZ = [int(np.ceil(dim/T)+1) for dim in [L, W, H]]
        
        padding = 0
        dX, dY, dZ = padding*L/sNX, padding*W/sNY, padding*H/sNZ
        sNX += padding*2
        sNY += padding*2
        sNZ += padding*2

        # Create a fine scale spacing based on this coarse sampling size
        # and the desired resolution
        N = int(np.ceil(1/self.elementSize))
        NX, NY, NZ = [int(n*N)+1 for n in [sNX, sNY, sNZ]]
        

        logger.debug(f"Discretizing with ({NX}, {NY}, {NZ}) points "
                     f"({NX*NY*NZ:,} total).")
        X, Y, Z = np.meshgrid(np.linspace(-L/2-dX, L/2+dX, NX),
                              np.linspace(-W/2-dY, W/2+dY, NY),
                              np.linspace(-H/2-dZ, H/2+dZ, NZ))
        grid = pv.StructuredGrid(X, Y, Z)
        points = np.vstack((X.ravel(order='F'), 
                            Y.ravel(order='F'), 
                            Z.ravel(order='F'))).T
        
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=".")

        with Timing("Evaluating SDF over grid", logger=logger):
            # Evaluate the sdf using a coarse sampling (on the order of 
            # the ligament/plate thickness) first
            if self.elementSize <= 0.5:
                # Create an index array for the meshed grid
                fullinds = np.arange(X.size).reshape((NY, NX, NZ), 
                                                      order='F')
                values = np.ones((X.size, 3))

                # Check that the sampling grid is finer than the coarse
                # sampling 
                if sNX == sNY == sNZ == 1:
                    values = func(points)
                    logger.debug("Coarse grid is the same as the full grid."
                                 " No further sampling required.")
                else:
                    # Create a resampling grid that is slightly larger than the geometry thickness
                    midX, midY, midZ = int(NX/2), int(NY/2), int(NZ/2)
                    subgrid = fullinds[midY-N-2:midY+N+2, 
                                    midX-N-2:midX+N+2, 
                                    midZ-N-2:midZ+N+2] - \
                                            fullinds[midY, midX, midZ]  
                    subgrid = subgrid.ravel(order='F')

                    logger.debug("Coarse sampling with "
                                f"{X[::N,::N,::N].transpose((1, 0, 2)).shape} "
                                f"points ({X[::N,::N,::N].size:,} total).")
                    
                    # Calculate the SDF on the coarse grid
                    subpoints = np.vstack((X[::N,::N,::N].ravel(order='F'), 
                                        Y[::N,::N,::N].ravel(order='F'), 
                                        Z[::N,::N,::N].ravel(order='F'))).T
                    svalues = func(subpoints)
                    # ax.scatter(subpoints[:, 0], subpoints[:, 1], subpoints[:, 2], marker="x")
                    # Determine locations to subsample
                    inds = svalues <= T*1.5

                    # Find neighboring points on the fine mesh
                    logger.debug("Finding neighboring points to interior points "
                                 "found during the coarse sampling process.")

                    subinds = np.unique((fullinds[::N,::N,::N].ravel(order='F')[inds[:, 0]][:, None] +\
                                        subgrid[None, :]).ravel(order='F'))
                    subinds = subinds[subinds > 0]
                    subinds = subinds[subinds < fullinds.size]
                    rinds = subinds
                    
                    # tree = spatial.KDTree(points)
                    # resample = tree.query_ball_point(subpoints[inds[:, 0], :], 
                    #                                 1.25*T,
                    #                                 workers=-1)

                    # Subsample the coarse cells that have interior points
                    # rinds = np.unique(np.hstack(resample))
                    logger.debug(f"Fine sampling over {rinds.size:,} points.")
                    resamplepoints = points[rinds, :]
                    values[rinds] = func(resamplepoints)
                    # ax.scatter(resamplepoints[:, 0], resamplepoints[:, 1], resamplepoints[:, 2], marker="o")
                    # values[rinds] = -1
            else:
                logger.debug("Fine sampling over all points.")
                values = func(points)
            grid['scalars'] = values
            # fig.show()

        # Return the field clipped at the geometry boundary
        with Timing("Clipping SDF field at a value of 0", logger=logger):
            clip = grid.clip_scalar(value=0.)
            
        # with Timing("Using build in (parallel) generation", logger=logger):
        #     points = func.generate(step=elementSize, 
        #                            bounds=((-L/2, -W/2, -H/2),
        #                                    (L/2, W/2, H/2)))

        return clip
        
    def exportImage(self, save=False, size=[800, 800]):

        # Create VTK visualization
        vis = self.visualizeVTK()

        # Plot the image without an image window
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(vis.extract_geometry().triangulate(), 
                         color="w", smooth_shading=True)
        plotter.add_axes(box=True)
        # plotter.set_background("w")

        # If the image is to be saved to file, export it
        if save:
            save = self.imageFilename
        
        img = plotter.show(screenshot=save, window_size=size, 
                           return_img=True, return_cpos=False)
        
        # Return the image data
        return img

    @timing(logger)
    def run(self, reuse=False, export=False):
        """ Run geometry generation 
        
        Keywords
        --------
        reuse: boolean (default=True)
            Specifies whether or not to rerun the geometry generation.
            If True, existing geometry will be used if it already
            exists, otherwise, the geometry will be recreated.
        export: boolean (default=False)
            Specifies whether or not to export a mesh of the geometry.
        
        Returns
        -------
        None
        """
        
        logger.info(f"Calculating geometric properties of {self}.")
        super().run(reuse=reuse)

        # Define absolute geometric parameters
        margin = 1.1
        L = self.DIMENSION*self.length
        W = self.DIMENSION*self.width
        H = self.DIMENSION*self.height
        T = self.DIMENSION*self.thickness
        elementSize = T*self.elementSize
        
        if reuse and self.processed:
            logger.info("Geometry already processed and reuse requested."
                        " Nothing processed.")
        else:
            # Calculate internal surface are
            with Timing("Calculating internal surface area", logger):
                func = self.sdf
                surfaceMesh = self._mesh(func, L, W, H, elementSize)
                relativeSurfaceArea = surfaceMesh.area/(2*(L*W + W*H + H*L))
            
            # Calculate relative density
            with Timing("Calculating relative density", logger):
                func = self.sdf & sdf.box([L, W, H])
                volumeMesh = self._mesh(func, L*margin, W*margin, H*margin, elementSize)
                relativeDensity = volumeMesh.volume/(L*W*H)

            # Write out calcualted properties to a json file
            properties = {"relativeDensity": relativeDensity,
                          "relativeSurfaceArea": relativeSurfaceArea}
            with self.propertiesFilename.open('w') as f:
                json.dump(properties, f)

            # mesh and save
            if export:
                logger.info("Exporting stl mesh")
                volumeMesh.export(self.geometryFilename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    design = SDFGeometry('Hexagonal honeycomb', 3, 3, 3, 
                         thickness=0.3, radius=0.1,
                         directory=Path(__file__).parent/Path("tests"),
                         elementSize=0.5, form="graph")
    # design = SDFGeometry('Schwarz', 1, 1, 1, 
    #                      thickness=0.1, radius=0.25,
    #                      directory=Path(__file__).parent/Path("tests"),
    #                      elementSize=0.2, form="walledtpms")
    design.run(reuse=False, export=True)
    # print(design.relativeDensity, design.relativeSurfaceArea)
    # design.postprocess()
    # # design.exportImage(save=True)
    # design.plot().show()
    # print(design)

    # design.elementSize = 0.25

    # img = design.exportImage(save=True)
    # import matplotlib.pyplot as plt
    # # plt.rcParams['image.cmap'] = 'seismic'
    # design.sdf.show_slice(y=0, w=200, h = 200, bounds=((-20, -20, -20), (20, 20, 20)))
    # # design.sdf.show_slice(z=15., w=100, h = 100, bounds=((-15, -15, -15), (15, 15, 15)))
    # ax = plt.gca()
    # children = ax.get_children()
    # im = [child for child in children if "image" in str(type(child))][0]
    # im.set_clim(-1, 1)
    # im.set_cmap('seismic')
    # ax.axvline(x=15)
    # ax.axvline(x=-15)
    # plt.pause(10)


    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    geom = design.visualizeVTK(1, 1, 1)
    # pr.disable()
    # pr.dump_stats("profile.profile")
    # tmp = geom.extract_geometry().triangulate()
    # tmp.decimate_pro(0.75).plot(show_edges=True)
    # geom.plot(smooth_shading=True, color='w')

    
