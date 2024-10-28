import numpy as np
import logging
from unitcellengine.utilities import timing, Timing
from unitcellengine.geometry.sdf import SDFGeometry
import numba as nb
import numpy.typing as npt
import typing
from pathlib import Path
import pyvista as pv

# Create logger
logger = logging.getLogger(__name__)

NODE_EL = 8

def nodeDofs(n: npt.ArrayLike, ndof: int=3) -> npt.NDArray[np.int_]:
    """ DOF numbers corresponding to the specified nodes 
    
    Arguments
    ---------
    n: list like
        List of N nodes to extract degree of freedom numbers from.
    
    Keywords
    --------
    ndof: int
        Number of degrees of freedom per node in mesh.
    
    Returns
    -------
    (ndof, N) numpy array mapping degrees of freedom in the rows for
    each node.

    Notes
    -----
    - This implementation assumes that all nodes in the mesh have the
      number of degrees of freedom and leverages this to create the
      mapping.
    """

    basedofs = np.array(n)*ndof
    dofs = np.vstack([basedofs]*ndof)
    for i in range(ndof):
        dofs[i, :] += i

    return dofs.astype(np.int64)

@timing(logger)
@nb.njit(cache=True)
def emapping(nelx: int, nely: int, nelz: int, nrhos: npt.NDArray[np.float64]) -> \
        typing.Tuple[npt.NDArray[np.uint64], npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """ Element/node mapping for a given voxel mesh 
    
    Arguments
    ---------
    nelx, nely, nelz: int
        Number of voxel elements in the x|y|z direction
    nrhos: numpy (N,) array
        Nodal densities (1 if in the body and 0 if out of the body)
    
    Returns
    -------
    e2n: (nel, nodes/element) numpy unsigned int array
        Hexahedral-based nodal definitions for each element in the mesh.
        Each row corresponds to an element definition and the columns
        define the nodes in the element using the standard CCW notation.
    rhos: (nel,) numpy float array
        The density of each element in the mesh.
    eind: (nel,) boolean array
        Defines the indices that are inside the geometry (True) and those
        that are not (False)
    nind: (nnode, ) boolean array
        Defines the nodes that are inside the geometry (True) and those
        that are not (False)
    
    Notes
    -----
    - This is for an 8 node hexahedral element
    - This mapping methodology assumes a specific form for the voxel:
        - Node 0 starts at the min x, y, z location.
        - Node numbering increments in the x direction until the end of 
          the row. It then increments in y and then starts from xmin again.
        - Once a voxel slice is complete, the z direction is increased
          and the node number method is continued.
    - The node order is defined as follows
        Bottom slice     Top slice
        3<-----2         7<-----6
        |      ^         |      ^
        |      |         |      |
        0----->1         4----->5
    """

    # Mesh properties
    nel = nelx*nely*nelz           # Number of elements
    nel_slice = nelx*nely          # Number of elements per voxel slice
    nn_slice = (1+nelx)*(1+nely)   # Number of nodes per voxel slice

    # Initialize relevant arrays
    e2n=np.zeros((nel, NODE_EL), dtype=np.uint64)
    rhos = np.zeros(nel)
    einds = np.zeros(nel, dtype=np.bool_)
    ninds = np.zeros((1+nelx)*(1+nely)*(1+nelz), dtype=np.bool_)
    nodes = np.zeros(8, dtype=np.int64)

    # Build the element/node mapping matrix used to define the mesh.
    for elx in range(nelx):
        for ely in range(nely):
            for elz in range(nelz):
                # Element ID
                el = elx + ely*nelx + elz*nel_slice

                # Base node in row 1, slice 1
                n1 = elx + (nelx+1)*ely + elz*nn_slice
                # Base node in row 2, slice 1
                n2 = elx + (nelx+1)*(ely+1)  + elz*nn_slice

                # Determine if element is within the geometry based on
                # the nodal densities. If any node has a positive
                # density, then the element is considered to be in the mesh.
                nodes[:4] = np.array([n1, n1+1, n2+1, n2])
                nodes[4:] = nodes[:4] + nn_slice
                subnrhos = nrhos[nodes]
                if np.any(subnrhos > 0.):
                    einds[el] = True
                    ninds[nodes] = True
                    rhos[el] = np.mean(subnrhos)


                # Define node indexing on slice 1 in a counter-clockwise direction
                e2n[el, :4] = np.array([n1, n1+1, n2+1, n2])

                # Define node indexing on slice 2 in a counter-clockwise direction
                e2n[el, 4:] = e2n[el, :4] + nn_slice
                
    return e2n, rhos, einds, ninds

@timing(logger)
def mesh(sdf: typing.Callable, elementSize: float=1, 
         dimensions: npt.ArrayLike=[1, 1, 1], 
         center: npt.ArrayLike=[0, 0, 0],
         filename: typing.Union[None, str, Path]=None) -> \
         typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.uint64], typing.List[float],
                      npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """ Hex mesh sign distance function over a regtangular grid 
    
    Arguments
    ---------
    sdf: callable function sdf((N, 3)) -> (N,)
        Sign distance function definition for the geometry, that takes
        in points in 3D space and outputs there distance from the
        nearest geometric surface (with negative values being interior
        to the geometry and positive being exterior)
    
    Keywords
    --------
    elementSize: float > 0 (default is 1)
        Defines the approximate voxel mesh grid spacing, nothing that
        the value gets slightly modified to ensure perfect spacing
        within the dimensions of the geometric bounds defined by
        *dimensions*.
    center: array-like of length 3 (default is [0, 0, 0])
        Defines the (x, y, z) centerpoint of the geometry.
    dimensions: array-like of length 3 (default is [1, 1, 1])
        Defines the (x, y, z) bounding box sizes.
    filename: None, str, or Path object
        If not None, defines the filename to save the mesh file to. Must
        either be a pkl (python Pickle file) or vtu (Paraview) format.
    
    Returns
    -------
    ns: (nnodes, 4) numpy array
        Node list in the form (nid, x, y, z)
    ninds: (nnodes,) boolean array
        Defines the node indices that reside within the geometry.
    e2n: (nel, 8) numpy array
        Defines the node mapping for each element in the mesh using a
        counterclockwise convention.
    nel: length 3 list
        Defines the number of elements in the x, y, and z directions.
    erhos: (nel,) numpy array
        Element densities (0 <= rho <= 1) where 0 defines an inactive
        element, 1 defines a fully active element, and intermediate
        values indicate a partially active element.
    einds: (nel,) boolean array
        Defines the element indices that reside within the geometry

    """

    # Convert inputs to relevant mesh properties
    L, W, H = dimensions
    xc, yc, zc = center
    xmin, xmax = -L/2+xc, L/2+xc
    ymin, ymax = -W/2+yc, W/2+yc
    zmin, zmax = -H/2+zc, H/2+zc
    nelx = int(L/elementSize)
    nely = int(W/elementSize)
    nelz = int(H/elementSize)
    nnodes = (nelx+1)*(nely+1)*(nelz+1)
    nodes = np.arange(nnodes)
    nn_slice = (1+nelx)*(1+nely)
    dx, dy, dz = L/nelx, W/nely, H/nelz

    # Generate the base voxel mesh. Note, to get the nodes laid out in
    # the correct orientation for element definition, the x, y, and z
    # inputs are arranged such that mesh grid outputs the results in the
    # desired format.
    X, Y, Z = np.meshgrid(np.linspace(xmin, xmax, nelx+1),
                          np.linspace(ymin, ymax, nely+1),  
                          np.linspace(zmin, zmax, nelz+1), indexing='ij')
    # Y, Z, X = np.meshgrid(np.linspace(ymin, ymax, nely+1),
    #                       np.linspace(zmin, zmax, nelz+1),  
    #                       np.linspace(xmin, xmax, nelx+1))
    
    # Flatten the voxel mesh and assign the coordinates to a node number
    xs = X.flatten('F')
    ys = Y.flatten('F')
    zs = Z.flatten('F')
    ns = np.vstack((nodes, xs, ys, zs)).T
    logger.debug(f"Total number of nodes in the voxel mesh: {ns.shape[0]}")

    # Calculate nodal sign distance function values
    with Timing("Calculating nodal SDF values", logger=logger):
        nsdf = sdf(ns[:, 1:]).flatten()
        nrhos = np.ones(nsdf.shape)
        nrhos[nsdf>0] = 0 

    # Create full element to node map
    e2n, rhosmean, einds, ninds = emapping(nelx, nely, nelz, nrhos)
    logger.debug(f"Total number of active nodes: {ninds.sum()}")
    logger.debug(f"Total number of active elements: {einds.sum()}")

    # Go through and implement a more accurate element density for
    # boundary elements (i.e., 0 < rhomean < 1) by integrating the 
    # sign distance field using Gauss quadrature. He, an 8 point
    # quatrature is used. Note: the weights for these quadrature points
    # are all 1 and are therefore omitted.

    with Timing("Intermediate density integration", logger=logger):
        # Update the density only at locations of intermediate density (to
        # improve efficiency while processing large meshes)
        inds = np.logical_and(einds, rhosmean < 1)
        if np.any(inds):
            subns = ns[e2n[inds, 0], 1:]
            gps = 0.5*(1+np.array([-1/np.sqrt(3), 1/np.sqrt(3)]))
            
            # Calculate the sign distance function at each integration point
            corners = np.hstack([-sdf(subns + np.array([dX, dY, dZ])) 
                                                        for dX in [0, dx] 
                                                        for dY in [0, dy] 
                                                        for dZ in [0, dz]])
            offsets = np.hstack([-sdf(subns + np.array([dX, dY, dZ])) 
                                                        for dX in dx*gps 
                                                        for dY in dy*gps
                                                        for dZ in dz*gps])
            
            # Combine the integration and nodal point data
            prhos = np.hstack((corners, offsets))
            # Sum up the integration points and normalize them by cell volume.
            # This is then the volume average distance from the surface in
            # absolute units. We then normalize with respect to the element
            # diagonal size, which gives us the mean nomalized distance. We then
            # offset this number by 0.5 to get a density of 0.5 when the mean
            # distance is zero (i.e., half the body is in the cell and the other 
            # half isn't)
            # NOTE: TPMS implicit representations aren't actually sign
            # distance functions (they are nonlinear in distance and have
            # incorrect magnitude.) We therefore have to normalize based on
            # the local information.
            normc1 = np.abs(corners[:, 1]-corners[:, 0])/dz
            normc2 = np.abs(corners[:, 2]-corners[:, 0])/dy
            normc3 = np.abs(corners[:, 4]-corners[:, 0])/dx
            normc4 = np.abs(corners[:, 7]-corners[:, 0])/np.sqrt(dx**2+dy**2+dz**2)
            scaling = max(normc1.max(), normc2.max(), 
                        normc3.max(), normc4.max())*np.sqrt(dx**2+dy**2+dz**2)
            erhos = rhosmean.copy()
            erhos[inds] = prhos.sum(axis=1)/16/scaling+0.5
            
            # Make sure there aren't any small negative densities
            erhos[erhos < 0.] = 0.

            # Store the integration point binary densities (which are used
            # to calculate stresses)
            # intrhos = np.zeros()
        else:
            # If there are no intermediate density elements, just use
            # the current element density definition
            erhos = rhosmean.copy()

    # Save the mesh if desired
    out = (ns, nsdf, ninds, e2n, [nelx, nely, nelz], erhos, einds)
    if filename:
        filename = Path(filename)

        # Parse the input suffix
        suffix = filename.suffix
        if suffix == "":
            suffixes = [".npz", ".vtu"]
        elif not (suffix == ".npz" or suffix == ".vtu"):
            logger.warning(f"Suffix {suffix} is an unsupported export "
                            "Exporting as '.pkl' and '.vtu' instead.")
            suffixes = [".npz", ".vtu"] 
        else:
           suffixes = [suffix] 
        
        # Write out files for each file type specified
        for suffix in suffixes:
            if suffix == ".npz":
                # Save as a python numpy array
                save = {k: v for k, v in 
                            zip(["nodes", "nsdf", "ninds", \
                                 "e2n", "nels", "erhos", "einds"],
                                out)}
                with Timing(f"Saving mesh to {filename.with_suffix('.npz')}",
                            logger=logger):
                    np.savez_compressed(filename.with_suffix(".npz"), **save)
            else:
                # Save as a paraview file
                nel = einds.sum()
                with Timing(f"Saving mesh to {filename.with_suffix('.vtu')}",
                            logger=logger):
                    grid = _convert2pyvista(ns, nsdf, e2n, erhos, einds, {"sdf": nsdf}, {"density": erhos[einds]})
                    grid.save(filename.with_suffix(".vtu"))
    return out

def periodic(nodes: npt.ArrayLike, e2n: npt.ArrayLike, nels: list) ->\
        typing.Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """ Return periodic element node mapping 
    
    Arguments
    ---------
    nodes: (N, 4) numpy array
        Defines all of the nodes in the mesh, including the node index
        (nid), and the (x, y, z) coordinate of the nodate: (nind, x, y,
        z)
    e2n: (nel, 8) numpy array
        Defines the node mapping for each element in the mesh based on a
        counter clockwise structure.
    nels: length 3 list
        Defines the number of elements in the x, y, and z directions
    
    Returns
    -------
    pe2n: (nel, 8) numpy array
        New the node mapping for each element with periodic boundaries
    lns: (M,) numpy array
        "Leading" periodic nodes such that *lns[i]* = *fns[i]*
    fns: (M,) numpy array
        "Following" periodic nodes such that *lns[i]* = *fns[i]*
     
    """

    # Pull out the number of elements in each direction
    nelx, nely, nelz = nels

    # Reshape the nodes into a 3D grid, relying on the known structure
    # of the grid to do so
    Ns = nodes[:, 0].reshape((nelx+1, nely+1, nelz+1), 
                              order='F').astype(np.int64)
    
    # Pull out the "leading" nodes (-X, -Y, and -Z faces)
    lns = np.hstack((# Faces (without edges)
                     Ns[0, 1:-1, 1:-1].flatten(), 
                     Ns[1:-1, 0, 1:-1].flatten(), 
                     Ns[1:-1, 1:-1, 0].flatten(),
                     # Edges (without end nodes)
                     Ns[0, 0, 1:-1].flatten(),
                     Ns[0, 0, 1:-1].flatten(),
                     Ns[0, 0, 1:-1].flatten(),
                     Ns[0, 1:-1, 0].flatten(),
                     Ns[0, 1:-1, 0].flatten(),
                     Ns[0, 1:-1, 0].flatten(),
                     Ns[1:-1, 0, 0].flatten(),
                     Ns[1:-1, 0, 0].flatten(),
                     Ns[1:-1, 0, 0].flatten(),
                     # Vertices
                     np.array([Ns[0, 0, 0]]*7),
                     ))
    # lns = np.hstack((Ns[0, :, :].flatten(), 
    #                    Ns[:, 0, :].flatten(), 
    #                    Ns[:, :, 0].flatten()))
    
    # Pull out the "following" nodes (+X, +Y, and +Z faces)
    fns = np.hstack((# Faces (without edges)
                     Ns[-1, 1:-1, 1:-1].flatten(), 
                     Ns[1:-1, -1, 1:-1].flatten(), 
                     Ns[1:-1, 1:-1, -1].flatten(),
                     # Edges (without end nodes)
                     Ns[0, -1, 1:-1].flatten(),
                     Ns[-1, 0, 1:-1].flatten(),
                     Ns[-1, -1, 1:-1].flatten(),
                     Ns[0, 1:-1, -1].flatten(),
                     Ns[-1, 1:-1, 0].flatten(),
                     Ns[-1, 1:-1, -1].flatten(),
                     Ns[1:-1, 0, -1].flatten(),
                     Ns[1:-1, -1, 0].flatten(),
                     Ns[1:-1, -1, -1].flatten(),
                     # Vertices
                     np.array([Ns[0, 0, -1],
                               Ns[0, -1, 0],
                               Ns[-1, 0, 0],
                               Ns[0, -1, -1],
                               Ns[-1, 0, -1],
                               Ns[-1, -1, 0],
                               Ns[-1, -1, -1]]),
                     ))
    # fns = np.hstack((Ns[-1, :, :].flatten(), 
    #                    Ns[:, -1, :].flatten(), 
    #                    Ns[:, :, -1].flatten()))
    
    # Ns[-1, :, :] = Ns[0, :, :]
    # Ns[:, -1, :] = Ns[:, 0, :]
    # Ns[:, :, -1] = Ns[:, :, 0]

    # Create a new element node mapping that wraps around periodically
    pns = Ns.flatten('F')
    pns[fns] = pns[lns]
    return pns[e2n], lns, fns

def _convert2pyvista(ns, nsdf, e2n, erhos, einds, pointData, cellData):
    """ Create a pyvista plotting object from the given raw mesh details 
    
    Parameters
    ----------
    ns: Nx4 numpy array
        Nodal coordinates for the mesh, with the first 1st column defining
        the node number and the last 3 columns defining the (x, y, z) coordinates.
    nsdf:
        Geometry signed distance field values at each node
    e2n: numpy array
        Element nodal connectivity mapping, with each row defining the nodal conductivity.
    erhos: numpy array
        Volume fraction of each element (percentage of the geometry that fills the element).
    einds: numpy array
        Index array defining the active elements in the mesh (within or on the boundary of the geometry).
    pointData: dict [Default = {}]
        Dictionary of nodal data, where the dictionary key defines the 
        vtk variable name and the value must be a number of nodes x dofs
        array.
    cellData: dict [Default = {}]
        Dictionary of cell data, where the dictionary key defines the
        vtk variable name and the value must be a number of cells x dofs
        array.

    Returns
    -------
    pyvista UnstructuredGrid object that contains mesh data (such as the geometry
    signed-distance function values and the element fill ration) and the specified
    nodal and cell data.

    """
    # Incorporate the basis mesh data
    pointData.update({"sdf": nsdf})
    cellData.update({"density": erhos[einds]})

    # Write results to file
    nel = einds.sum()
    with Timing(f"Creating pyvista representation of mesh {mesh}", logger=logger):
        # Define the point connectivity for each cell
        cells = np.hstack((np.ones((nel, 1), dtype=int)*e2n.shape[1], e2n[einds, :].astype(int)))
        # Define the cell type for each cell in the mesh (which are all hexahedron)
        # @TODO This is a fragile way to determine the cellType based on the node connectivity matrix.
        cellType = pv.CellType.HEXAHEDRON if e2n.shape[1] == 8 else pv.CellType.QUADRATIC_HEXAHEDRON
        cellTypes = np.full(nel, cellType, dtype=np.uint8)

        # Create the pyvista grid object and add in the supplied data
        grid = pv.UnstructuredGrid(cells.ravel(), cellTypes, ns[:, 1:])
        for k, v in pointData.items():
            grid.point_data[k] = v
        for k, v in cellData.items():
            grid.cell_data[k] = v

    return grid

def convert2pyvista(mesh: str|Path, pointData:dict={}, cellData: dict={}) -> pv.UnstructuredGrid:
    """ Create a pyvista plotting object for the given mesh and data

    Parameters
    ---------
    mesh: str or Path object with .npz extension
        Filename of mesh file generated by internal homogenization
        engine.
    pointData: dict [Default = {}]
        Dictionary of nodal data, where the dictionary key defines the 
        vtk variable name and the value must be a number of nodes x dofs
        array.
    cellData: dict [Default = {}]
        Dictionary of cell data, where the dictionary key defines the
        vtk variable name and the value must be a number of cells x dofs
        array.

    Returns
    -------
    pyvista UnstructuredGrid object that contains mesh data (such as the geometry
    signed-distance function values and the element fill ration) and the specified
    nodal and cell data.

    """

    # Process mesh filename
    mesh = Path(mesh)
    assert mesh.suffix == ".npz", (
        f"Input mesh {mesh} has the incorrect " "suffix. Should be .npz."
    )

    # Load mesh
    with np.load(mesh) as data:
        ns = data["nodes"]
        nsdf = data["nsdf"]
        e2n = data["e2n"]
        erhos = data["erhos"]
        einds = data["einds"]

    grid = _convert2pyvista(ns, nsdf, e2n, erhos, einds, pointData, cellData)

    return grid

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    from pathlib import Path

    geometry = SDFGeometry("Face centered cubic", 1, 1, 1, 0.3, 
                           radius=0.25, form="graph")
    # geometry = SDFGeometry("Gyroid", 5, 5, 5, 0.05, form="walledtpms")
    # geometry.run(reuse=False)
    

    L = geometry.DIMENSION*geometry.length
    W = geometry.DIMENSION*geometry.width
    H = geometry.DIMENSION*geometry.height
    T = geometry.DIMENSION*geometry.thickness

    dims = [L, W, H]
    elementSize = 0.05*T
    filename = Path(__file__).parent/Path("tests")/Path("test.npz")
    ns, nsdf, ninds, e2n, (nelx, nely, nelz), rhos, einds = \
        mesh(geometry.sdf, elementSize=elementSize, dimensions=dims, 
             filename=filename)
    # ns, nsdf, ninds, e2n, (nelx, nely, nelz), rhos, einds = mesh(geometry.sdf, elementSize=elementSize, dimensions=dims, filename=None)
    grid = convert2pyvista(filename.with_suffix(".npz"))
    grid.plot()
    # print(nsdf.min())
    # print(geometry.relativeDensity)
    print("done")
