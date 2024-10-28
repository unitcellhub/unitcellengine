from pathlib import Path
import numpy as np
from unitcellengine.geometry import Geometry
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

def mesh(geometry, elementSize):
    """ Create triply periodic mesh of unit cell geometry """

    # Check that the geometry file exists
    geometry = Path(geometry)
    if not geometry.exists():
        text = f"Geometry file {geometry} doesn't exist. Couldn't " +\
                "mesh geometry"
        logger.critical(text)
        raise IOError(text)
    
    logger.info(f"Meshing geometry file {geometry}.")

    # Initialize gmsh
    _gmsh.initialize()

    # Allow gmsh print output based on logger level
    # if logger.level == logging.DEBUG:
    _gmsh.option.setNumber("General.Terminal", 1)
    
    # # Load geometry
    # if 'msh' in geometry.suffix:
    #     # Load the mesh file, which contains native gmsh geometry
    #     # definition and is the preferred file format.
    #     _gmsh.merge(geometry.as_posix())

    #     # # Check to see if physical names have been defined
    #     # surfaces = [s[1] for s in _gmsh.model.getEntities(2)]
    #     # groups = _gmsh.model.getPhysicalGroupsForEntity(2, surfaces)
    #     # flagGroups = False
    #     # expected = {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'interior'}
    #     # check = expected.intersection(set(groups))
    #     # if len(check) != len(expected):
    #     #     flagGroups = False
    #     #     if len(check) > 0:
    #     #         difference = expected.difference(check)
    #     #         logger.info(f"Physical groups [{', '.join([f'{g}' for g in check])}] "
    #     #                     "already exist, but missing "
    #     #                     f"[{', '.join([f'{g}' for g in difference])}]. "
    #     #                     "Flagging for additional group processing.")
    # else:
    #     # Load the geometry with the OpenCASCADE importer
    volumes = _gmsh.model.occ.importShapes(geometry.as_posix())
    for volume in volumes:
        _gmsh.model.occ.healShapes([volume])
    _gmsh.model.occ.removeAllDuplicates()
    _gmsh.model.occ.synchronize()
    

    # Ask OpenCASCADE to compute more accurate bounding boxes of
    # entities.
    _gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)

    # Pull out geometry boundaries and sizes
    vbounds = np.array([_gmsh.model.occ.getBoundingBox(*v) for v in volumes])
    xmin, ymin, zmin = vbounds[:, :3].min(axis=0).tolist() 
    xmax, ymax, zmax = vbounds[:, 3:].max(axis=0).tolist()

        
    boundaries = dict(xmin=xmin, ymin=ymin, zmin=zmin,
                      xmax=xmax, ymax=ymax, zmax=zmax)
    dims = dict(x=xmax-xmin, y=ymax-ymin, z=zmax-zmin)

    
    # Define function to finding matching surfaces
    eps = 1e-2
    signs = [-1]*3 + [1]*3
    def matching():
        matches = dict(x=[], y=[], z=[])
        for i, dim in enumerate('xyz'):
            # Create the baseline bounding box
            inputs = [boundaries[bound]+eps*sign for bound, sign in zip(
                                                        ['xmin', 'ymin', 'zmin', 
                                                        'xmax', 'ymax', 'zmax'],
                                                        signs)]
            # Update the bounding box to slice the relevant min boundary only
            inputs[i+3] = boundaries[dim+'min']+eps

            # Pull out the surfaces within that bounding box slice
            tmins = _gmsh.model.getEntitiesInBoundingBox(*inputs, 2)

            # Make sure surfaces are found. If not, throw a warning
            if len(tmins) == 0:
                logger.warning(f"No surfaces found on the {dim} min boundary. "
                                "This may be a feature of the unit cell, but "
                                "should be verified.")
                continue

            # For each identified surface, pull out the corresponding 'max'
            # surface 
            flagMatch = False
            for tmin in tmins:
                # Get the bounding box for the surface of interest
                bmin = _gmsh.model.getBoundingBox(*tmin)
                
                # Translate the bounding box to the other size of the unitcell
                # and pull out the corresponding surface
                inputs = [bound+eps*sign for bound, sign in zip(bmin, signs)]
                inputs[i] += dims[dim]
                inputs[i+3] += dims[dim]
                tmaxes = _gmsh.model.getEntitiesInBoundingBox(*inputs, 2)

                # Check that surfaces were found on the opposing face
                if len(tmaxes) == 0:
                    text = "Unable to find any surfaces opposing " +\
                            f"surface {tmin[1]} in the {dim} direction."
                    logger.critical(text)
                    raise RuntimeError(text)
                    
                if len(tmaxes) > 1:
                    text = "Multiple matching surfaces were found for " +\
                            f"surface {tmin[1]} in the {dim} direction. " +\
                            "This is unexpected behavior. Verify that the " +\
                            "geometry is valid and doesn't contain " +\
                            "duplicate surfaces."
                    logger.critical(text)
                    raise RuntimeError(text)

                # For all the matches, we compare the corresponding bounding boxes...
                tmax = tmaxes[0]
                # Pull out the bounds of the identifies surface
                bmax = _gmsh.model.getBoundingBox(*tmax)

                # Translate the bounds back to the min size of the unit cell
                bmax = list(bmax)
                bmax[i] -= dims[dim]
                bmax[i+3] -= dims[dim]

                # ...and if they match, we apply the periodicity constraint
                xmin, ymin, zmin, xmax, ymax, zmax = bmin
                xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = bmax
                bmax = _gmsh.model.getBoundingBox(*tmax)
                if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                        and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                        and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                    logger.info(f"Dim {dim}: Linked surface {tmin[1]} with "
                                f"{tmax[1]}.")
                    matches[dim].append((tmin[1], tmax[1]))
                else:
                    # something went wrong
                    text = "Something went wrong will matching periodic " +\
                           f"faces in the {dim} direction. It seems as " +\
                           f"surfaces {tmin[1]} and {tmax[1]} match, but " +\
                            "their bounding boxes don't quite line up. " +\
                            "Try updating the matching tolerance."
                    logger.critical(text)
                    raise RuntimeError(text)
        return matches


    # Enforce periodicity across the unit cell

    # Pull out all the surfaces on the "min" side of the unit cell and link
    # them to the "max" side
    # boundaries = dict(xmin=-L2, xmax=L2,
    #                 ymin=-W2, ymax=W2,
    #                 zmin=-H2, zmax=H2)
    # dims = dict(x=L, y=W, z=H)
    periodic = matching()
    for i, dim in enumerate('xyz'):
        # For each periodic face set, imprint geometries where necessary
        # to ensure matching geometry definitions
        for face1, face2 in periodic[dim]:
            tmin = (2, face1)
            tmax = (2, face2)

            # Get the bounding box for the surface of interest
            bmin = _gmsh.model.getBoundingBox(*tmin)
            bmax = _gmsh.model.getBoundingBox(*tmax)

            # Pull out the surface points
            boxmin = [bound+eps*sign for bound, sign in zip(bmin, signs)]
            boxmax = [bound+eps*sign for bound, sign in zip(bmax, signs)]
            pmin = _gmsh.model.getEntitiesInBoundingBox(*boxmin, 0)
            pmax = _gmsh.model.getEntitiesInBoundingBox(*boxmax, 0)
            cmin = _gmsh.model.getEntitiesInBoundingBox(*boxmin, 1)
            cmax = _gmsh.model.getEntitiesInBoundingBox(*boxmax, 1)

            # Pull out the coordinates of each point on the periodic
            # surfaces
            coordmin = [_gmsh.model.getValue(p[0], p[1], [0]) for p in pmin]
            coordmax = [_gmsh.model.getValue(p[0], p[1], [0]) for p in pmax]
            
            # Convert the coordinates to each surfaces parameterization
            uvmins = [_gmsh.model.getParametrization(2, tmin[1], c) 
                                                    for c in coordmin]
            uvmaxes = [_gmsh.model.getParametrization(2, tmax[1], c) 
                                                    for c in coordmax]
            
            # Identify points that exist on one surface, but
            # not the other. Group them accordingly.
            nuvmins = []
            nuvmaxes = []
            for uvs, uvrefs, new in zip([uvmins, uvmaxes], 
                                        [uvmaxes, uvmins], 
                                        [nuvmaxes, nuvmins]):
                for uv in uvs:
                    check = np.isclose(np.array([uv]*len(uvrefs)), 
                                    np.array(uvrefs)).sum(axis=1) == 2
                    if check.sum() == 0:
                        new.append(uv) 
            
            
            # Loop over each matching surface and impose
            # periodic geometry, splitting curves where
            # necessary. 
            for surface, new, curves in zip([tmin, tmax], 
                                        [nuvmins, nuvmaxes],
                                        [cmin, cmax]):
                points = []
                # Loop over each missing point
                # that is missing on this surface and add
                # it. 
                for uv in new:
                    coords = _gmsh.model.getValue(2, surface[1], uv)
                    
                    # Find the right curve to reparameterize
                    flagReparam = False
                    for c in curves:
                        # Pull out the parametric bounds for
                        # the curve
                        bounds = _gmsh.model.getParametrizationBounds(*c)
                        
                        # Discretize the curve to determine
                        # if the current point lies on this
                        # curve 
                        ts = np.linspace(*bounds, 1000)
                        xyzs = _gmsh.model.getValue(*c, ts).reshape(-1, 3)
                        
                        # Check the distance between the
                        # current point of interest and all
                        # points on the discretized curve 
                        length = np.linalg.norm(xyzs[1:, :]-xyzs[:-1, :], axis=1).sum()
                        distance = np.linalg.norm(coords-xyzs, axis=1)
                        
                        # If the point is close to the
                        # curve, assume the point is on the
                        # curve 
                        if (distance < length/len(ts)).any():
                            # Point is on curve
                            t = ts[distance.argmin()]
                            
                            # Create a new point at this
                            # location and split the surface
                            # at this point
                            # uv = _gmsh.model.reparametrizeOnSurface(*c, t, surface[1])
                            # coords = _gmsh.model.getValue(*c, t)
                            p = _gmsh.model.occ.addPoint(*coords)
                            # _gmsh.model.occ.synchronize()
                            _gmsh.model.occ.fragment(volumes, 
                                                    [(0, p)],
                                                    removeObject=True,
                                                    removeTool=True)
                            
                            flagReparam = True
                            break
                    assert flagReparam

    # Now that all of the periodic faces have been reparameterized to
    # match exactly, synchronize the (which will updated the surface
    # numbering) and update the periodic surface matches
    _gmsh.model.occ.synchronize()
    periodic = matching()    

    for i, dim in enumerate('xyz'):
        # Define the periodicity through a 4x4 affine matrix, which is
        # flattened into a list. Note that, the last column corresponds to
        # the translation element and is the only component that should be
        # modified 
        translate = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        translate[i*4+3] = dims[dim]

        # Apply periodicity
        fmin = [f for f, _ in periodic[dim]]
        fmax = [f for _, f in periodic[dim]]
        _gmsh.model.mesh.setPeriodic(2, fmax, fmin, 
                                            translate)

    # Group surfaces based on their geometry location. Note, we do this
    # after the periodic linking since the mesh matching procedure
    # affects the surface definitions
    physicalNames = dict(xmin=[], xmax=[], ymin=[], ymax=[],
                         zmin=[], zmax=[], interior=[])
    for surface in _gmsh.model.getEntities(2):
        # Get the surface bounding box
        box = _gmsh.model.getBoundingBox(*surface)
        
        # If any of the extremes are essentially the same, check to
        # see if they are on the boundary
        check = [np.isclose(box[i], box[i+3], atol=eps) for i in range(3)]
        if any(check):
            # Pull out the relevant boundary zero width
            i = np.array([0, 1, 2])[check]
            assert len(i) == 1
            i = i[0]
            dim = 'xyz'[i]

            # Determine if the surface is on the min or max surface
            for extreme, offset in zip(['min', 'max'], [0, 3]):
                if np.isclose(box[i+offset], boundaries[dim+extreme], 
                              atol=eps):
                    logger.debug(f"Surface {surface[1]} found to be "
                                    f"on the {dim[0]} {extreme} boundary.")
                    physicalNames[dim+extreme].append(surface)
                    continue 
        
        # If the surface is on the exterior, than it is an interior
        # surface 
        logger.debug(f"Surface {surface[1]} found to be "
                        "an interior surface.")
        physicalNames['interior'].append(surface)
    
    # Specify physical names for each surface grouping
    for name, surfaces in physicalNames.items():
        tags = _gmsh.model.addPhysicalGroup(2, surfaces)
        _gmsh.model.setPhysicalName(2, tags, name)

    # Set the mesh size
    volumes = _gmsh.model.getEntities(3)
    ps = _gmsh.model.getBoundary(volumes, False, False, True)  # Get all points
    _gmsh.model.mesh.setSize(ps, elementSize)

    # Create a physical name for all elements
    tags = _gmsh.model.addPhysicalGroup(3, volumes)
    _gmsh.model.setPhysicalName(3, tags, "all")

    # Mesh the geometry
    _gmsh.model.mesh.generate(3)

    # # Create 2nd order tets
    # _gmsh.model.mesh.setOrder(2)

    # # Delete surface mesh
    # # Note that 3 node tri elements are of type 2 and 9 node tri
    # # elements are of type 9
    # dims, etags, _ = _gmsh.model.mesh.getElements(2)
    # dimTags = [(2, tag) for tag in etags[0]]
    # _gmsh.model.mesh.clear(etags)
    

    # _gmsh.fltk.run()

    # Save the mesh in 2 formats: .msh and .inp
    # _gmsh.option.setNumber("Mesh.SaveAll", 1)
    # _gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
    # _gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
    _gmsh.write(geometry.with_suffix('.msh').as_posix())
    _gmsh.write(geometry.with_suffix('.inp').as_posix())
    _gmsh.write(geometry.with_suffix('.bdf').as_posix())
    # _gmsh.write(geometry.with_suffix('.bdf').as_posix())
    # _gmsh.write(geometry.with_suffix('.dat').as_posix())

    # Close out gmsh
    _gmsh.finalize()

if __name__ == "__main__":
    mesh(Path("unitcell/geometry/tests/unitcellGeometry.step"), 0.5)



