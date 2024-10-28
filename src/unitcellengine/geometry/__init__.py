import json
import plotly.graph_objects as go
from pathlib import Path
import logging
from unitcellengine.utilities import timing, cachedProperty
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import trimesh

# Create logger
logger = logging.getLogger(__name__)

# Default values
DEFAULT_THICKNESS = 0.3
DEFAULT_ELEMENT_SIZE = 0.2
DEFINITION_FILENAME = Path("unitcellDefinition.json")
PROPERTIES_FILENAME = Path("unitcellProperties.json")

class Geometry(object):
    """ Abstract unitcell classed for geometry generation """

    _GRAPH_UNITCELLS = []
    _WALLED_TPMS_UNITCELLS = []
    _geometryExtension = None
    _propertiesExtension = None

    DIMENSION = 10

    def __init__(self, unitcell, length, width, height, 
                 thickness=DEFAULT_THICKNESS,  radius=0.,
                 directory=".", elementSize=DEFAULT_ELEMENT_SIZE, 
                 form=None):
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
        form: None, "graph", or "walled tpms" (Default=None)
            Defines the unitcell form. If None, the form is
            automatically determined based on the *unitcell* name.
        
        """

        # Check subclass definition has been properly defined
        if not self._geometryExtension:
            raise NotImplementedError("Geometry file extension must "
                                      "be specified in the subclass "
                                      "definition: _geometryExtension")
        # if not self._GRAPH_UNITCELLS:
        #     raise NotImplementedError("Geometry graph unit cell types "
        #                               "must be specified in the subclass "
        #                               "definition: _GRAPH_UNITCELLS")
        # if not self._WALLED_TPMS_UNITCELLS:
        #     raise NotImplementedError("Geometry TPMS unit cell types "
        #                               "must be specified in the subclass "
        #                               "definition: _WALLED_TPMS_UNITCELLS")

        # Initialize properties
        self._unitcell = unitcell
        self.length = length
        self.width = width
        self.height = height
        self.directory = directory

        # Parse the unit cell type
        # If the unitcell form is set to auto (i.e. form=None) and the
        # unitcell name is found under multiple forms, throw a warning
        if form == None and unitcell in self._GRAPH_UNITCELLS and \
            unitcell in self._WALLED_TPMS_UNITCELLS:
            logger.warning(f"The unit {unitcell} was found within "
                            "multiple cell forms. The first found occurrence "
                            "will be used. To avoid this overlap, ",
                            "specify the 'form' option in the initialization.")
        
        # Convert 'form' to lowercase and remove spaces
        try:
            form = form.lower().replace(' ', '')
        except AttributeError:
            pass

        # Set unitcell parameters based in name and form
        if unitcell in self._GRAPH_UNITCELLS and form in [None, "graph"]:
            self._unitcells = self._GRAPH_UNITCELLS
            self._cellForm = "graph"
        elif unitcell in self._WALLED_TPMS_UNITCELLS and form in [None, 
                                                                 "walledtpms",
                                                                 "walled tpms"]:
            self._unitcells = self._WALLED_TPMS_UNITCELLS
            self._cellForm = "walledtpms"
        elif form and (unitcell in self._GRAPH_UNITCELLS+self._WALLED_TPMS_UNITCELLS):
            # It seems as those the specified form is invalid
            raise ValueError(f"Form {form} is invalid for unitcell "
                             f"{unitcell}. Try setting 'form' to None "
                             "for auto selection or updating this "
                             "to the valid form type.")
        else:
            options = self._GRAPH_UNITCELLS + self._WALLED_TPMS_UNITCELLS
            raise ValueError(f"Unitcell {unitcell} is invalid. Must be "
                             f"one of [{', '.join(options)}]")
        
        self.thickness = thickness
        self.radius = radius
        self.cache = True


    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.unitcell}, " +\
               f"L={self.length:.3f}, " +\
               f"W={self.width:.3f}, H={self.height:.3f}, " +\
               f"T={self.thickness:.3f}, form={self._cellForm})"
    
    @property
    def unitcell(self):
        """ Unit cell name """
        return self._unitcell
    
    @property
    def form(self):
        """ Unit cell form name (graph or walledtpms """
        return self._cellForm

    @property
    def processed(self):
        """ Has the design been processed? """
        try:
            if self.geometryFilename.exists() and self.propertiesFilename.exists():
                return True
            else:
                return False
        except:
            return False
    
    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, value):
        # Check to see if directory exists
        value = Path(value)
        # if not value.is_absolute():
        #     value = Path.cwd() / value
        
        value.mkdir(exist_ok=True)

        self._directory = value
    
    @property
    def definitionFilename(self):
        """ Filename of the JSON file that defines the input parameters """
        return self.directory / DEFINITION_FILENAME
    
    @property
    def geometryFilename(self):
        """ Filename of the generated geometry"""
        return self.directory / Path(f"unitcellGeometry.{self._geometryExtension}")

    @property
    def imageFilename(self):
        """ Filename of the generated geometry image"""
        return self.directory / Path(f"unitcellGeometry.png")

    @property
    def propertiesFilename(self):
        """ Filename of the generated unit cell properties """
        return self.directory / PROPERTIES_FILENAME
    
    @property
    def length(self):
        """ Normalized unit cell box length """
        return self._length
    
    @length.setter
    def length(self, value):
        assert value > 0., \
            f"Unit cell length should be greater than 0, not {value}"
        self._length = value
    
    @property
    def width(self):
        """ Normalized unit cell box width """
        return self._width
    
    @width.setter
    def width(self, value):
        assert value > 0., \
            f"Unit cell width should be greater than 0, not {value}"
        self._width = value
    
    @property
    def height(self):
        """ Normalized unit cell box height """
        return self._height
    
    @height.setter
    def height(self, value):
        assert value > 0., \
            f"Unit cell height should be greater than 0, not {value}"
        self._height = value
    
    @property
    def thickness(self):
        """ Normalized lattice ligament thickness """
        return self._thickness
    
    @thickness.setter
    def thickness(self, value):
        assert value > 0, \
            f"Ligament/wall thickness should greater than 0, not {value}"
        self._thickness = value
    
    @property
    def radius(self):
        """ Normalized joint smoothing radius """
        return self._radius
    
    @radius.setter
    def radius(self, value):
        assert value >= 0, \
            f"Smoothing radius greater than or equal to 0, not {value}"
        self._radius = value

    @property
    def relativeDensity(self):
        """ Unitcell relative density (read only)
        
        Note: this is the material volume relative to the unitcell
        volume 
        """
        if self.processed:
            return self.properties['relativeDensity']
        else:
            text = "Geometry has not been processed yet for " +\
                    f"{self}. Execute the 'run' method and " +\
                    "requery the relative density."
            logger.critical(text)
            raise AttributeError(text)

    @property
    def relativeSurfaceArea(self):
        """ Unitcell relative surface area (read only)
        
        Note: this is the internal surface area relative to the external
        unitcell area
        """
        if self.processed:
            return self.properties['relativeSurfaceArea']
        else:
            text = "Geometry has not been processed yet for " +\
                    f"{self}. Execute the 'run' method and " +\
                    "requery the relative surface area."
            logger.critical(text)
            raise AttributeError(text)

    @timing(logger)
    def run(self, reuse=True, blocking=True):
        """ Run lattice geometry generation
        
        Keywords
        --------
        reuse: boolean (default=True)
            Specifies whether or not to rerun the geometry generation.
            If True, existing geometry will be used if it already
            exists, otherwise, the geometry will be recreated.
        blocking: boolean (default=True)
            Specifies whether or not nTop should block further execution
            (True) or should run in the background (False). If run in
            the background, the process status can be monitored by the
            returned Popen object. (Deprecated)
        
        Notes
        -----
        - This superclass implementation should be run at the beginning of 
          each subclass implementation.
        - The subclass method needs to minimally calculate and write
          the unit cell relative density ("relativeDensity" property)
          and the unit cell relative surface area ("relativeSurfaceArea
          property) to the properties json file.

        """

        # Write geometry definition to file
        definition = {k: getattr(self, k) for 
                        k in ["unitcell", "length", "width", "height", 
                            "thickness", "radius", "elementSize", "form"]}
        with self.definitionFilename.open('w') as f:
            json.dump(definition, f)

        

    @cachedProperty
    def properties(self):
        """ Unit cell computed properties """
        properties = {}
        try:
            # Read unit cell propreties data
            with self.propertiesFilename.open('r') as f:
                properties = json.load(f)

        except FileNotFoundError:
            raise ValueError(f"Design {self} not run or failed during run.")

        return properties

    def exportImage(self, save=False, size=[800, 800]):
        """ Create an isographic image of the geometry 
        
        Keywords
        --------
        save: Boolean (default=False)
            Specify whether or not to save the file
        size: len 2 list (default=[800, 800])
            Defines the image size [width, height] in pixels

        Returns
        -------
        [size[0] x size[1] x 4] 8-bit rgba image data

        """
        raise NotImplementedError()

    def plot(self):
        """ Plot unit cell geometry """
        
        raise NotImplementedError()

class OversizedSTLGeometry(Geometry):
    """ Abstract class that generates oversized geometry and cuts it to size """

    _geometryExtension = "stl"


    @property
    def oversizedGeometryFilename(self):
        """ Filename of the generated geometry"""
        return self.directory / Path(f"unitcellGeometry_oversized.{self._geometryExtension}")

    @timing(logger)
    def postprocess(self, cleanup=True, reuse=False,
                    engines=['blender', 'scad']):
        """ Trim down the geometry to the specified dimensions """
        # requires, rtree, networkx, shapely
        # Install rtree with conda
        # Restall others with pip

        # If a rerun isn't desired and the geometry has already been
        # processed, exit early
        if reuse and self.processed:
            logger.info("Geometry already exists and reuse specified."
                        " No postprocessing needed.")
            return

        # Check the stl file
        stlOversized = self.oversizedGeometryFilename
        
        # Load the mesh with trimesh
        mesh = trimesh.load(stlOversized)
        
        # # Make sure the mesh is watertight. If it is not, fill in the
        # # holes. If the holes can't be filled, throw an error 
        # if not mesh.is_watertight:
        #     trimesh.repair.fill_holes(mesh)
        # if not mesh.is_watertight:
        #     text = f"The generated STL file {stlOversized} is not " +\
        #             "watertight and could not be fixed. This usually " +\
        #             "happens when the element size is too small or too " +\
        #             "big. The " +\
        #             f"current size is {self.elementSize}. Try something " +\
        #             "bigger or smaller."
        #     logger.error(text)
        #     raise RuntimeError(text)

        # Intersect the oversized unitcell with the unitcell box to get 
        # the final geometry. Try using the Blender tool first (which 
        # is fast). If that doesn't work, revert to the OpenSCAD tool, 
        # which is much slower, but much more robust.
        L, W, H = (self.DIMENSION*dim for 
                        dim in [self.length, self.width, self.height])
        bounds = dict(xmin=-L/2, xmax=L/2, ymin=-W/2, ymax=W/2, 
                      zmin=-H/2, zmax=H/2)
        box = trimesh.creation.box([L, W, H])
        for engine in engines:
            logger.info("Attempting to cut down geometry with the "
                        f"{engine} engine.")
            tmp = trimesh.boolean.intersection([box, mesh], engine=engine)
            if tmp.is_watertight:
                logger.info("Successfully cut down unit cell geometry.")
                break
        mesh = tmp

        # # Clean up the mesh
        # mesh.merge_vertices() 
        # mesh.remove_degenerate_faces()
        # trimesh.repair.fill_holes(mesh)

        # for i, dim in enumerate('xyz'):
        #     for sign, location in zip([1, -1], ['min', 'max']):
        #         # Define plane normal
        #         plane = [0, 0, 0]
        #         plane[i] = sign

        #         # Define cut plane location
        #         origin = [0, 0, 0]
        #         origin[i] = bounds[dim+location]

        #         # Make sure the mesh is watertight. If not, try to fill
        #         # it in
        #         # if not mesh.is_watertight:
        #         #     # mesh.merge_vertices() 
        #         #     # mesh.remove_degenerate_faces(tol)
        #         #     trimesh.repair.fill_holes(mesh)
        #         # assert mesh.is_watertight

        #         # Cut the mesh and fill in the opening
        #         if location == "min":
        #             bound = mesh.bounds[0]
        #         else:
        #             bound = mesh.bounds[1]
        #         if sign*origin[i] < sign*bound[i]:
        #             # Check that the cut plane intersects the body. If
        #             # not, skip cut step
        #             pass
        #         else:
        #             mesh = trimesh.intersections.slice_mesh_plane(mesh, plane, 
        #                                                       origin, cap=False,
        #                                                       process=True)
        #         # except AttributeError:
        #         #     # There might not be an intersection between the
        #         #     # the body and the plane.
        #         #     pass
        
        # # Merge all new mesh nodes, which seems to be required prior to
        # # running the mesh repair functions
        # mesh.merge_vertices() 
        # mesh.remove_degenerate_faces()

        # Check that the mesh is watertight one more time
        if not mesh.is_watertight:
            text = f"The modified STL file {stlOversized} is not " +\
                    "watertight. This usually " +\
                    "happens when the element size is too small or too " +\
                    "big. The " +\
                    f"current size is {self.elementSize}. Try something " +\
                    "bigger or smaller."
            logger.error(text)
            raise RuntimeError(text)
        
        # For some reason, the cap feature on the slice_mesh_plane
        # options can built faces with normals that point in the wrong
        # direction. So, we need to fix them here. 
        # logger.info("Repairing surface normals.")
        # trimesh.repair.fix_normals(mesh)

        # Save cutdown mesh
        mesh.export(self.geometryFilename)

        # Export image of geometry
        self.exportImage(save=True)

        # Cleanup files
        if cleanup:
            # Remove intermediate files
            logger.info("Cleaning up intermediate files. If this is not desired, "
                        "set the 'cleanup' option to False.")
            stlOversized.unlink()
            logger.info(f"Deleted {stlOversized}")
        else:
            logger.info("nTop intermediate files were not cleaned up because "
                        "'cleanup = False'. Set to True in the function call "
                        "if file cleanup is desired.")

    def exportImage(self, save=False, size=[800, 800]):

        # Open stl file and process relevant geometric properties
        filename = self.geometryFilename
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename.as_posix())
        
        # Define mapper function for poly data
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(reader.GetOutput())
        else:
            mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Determine the appropriate camera distance
        bounds = mapper.GetBounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        xc, yc, zc = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
        maxExtent = max(bounds)
        margin = 1.1
        fov = 45.                           # in degrees
        minDistance = maxExtent*margin/np.sin(0.5*fov*np.pi/180.)

        # Define camera properties
        camera = vtk.vtkCamera ()
        camera.SetViewAngle(fov)
        zheight = minDistance*np.sin(np.radians(35.264))
        camera.SetPosition(minDistance, minDistance, zheight)
        camera.SetFocalPoint(xc, yc, zc)

        # Create a rendering window and renderer
        ren = vtk.vtkRenderer()
        ren.SetActiveCamera(camera)
        renWin = vtk.vtkRenderWindow()
        renWin.OffScreenRenderingOn()
        renWin.AddRenderer(ren)
        renWin.SetAlphaBitPlanes(1)     # Enable usage of alpha channel
        renWin.SetSize(*size)
        ren.SetBackground(1, 1, 1)   # Background color white

        # Assign actor to the renderer
        ren.AddActor(actor)

        # Create screenshot
        # https://vtk.org/Wiki/VTK/Examples/Cxx/Utilities/Screenshot
        renWin.Render()
        imageFilter = vtk.vtkWindowToImageFilter()
        imageFilter.SetInput(renWin)
        imageFilter.SetInputBufferTypeToRGBA()
        imageFilter.ReadFrontBufferOff()
        imageFilter.Update()

        # Pull out image data
        img = imageFilter.GetOutput()
        width, height, _ = img.GetDimensions()
        vtkArray = img.GetPointData().GetScalars()
        components = vtkArray.GetNumberOfComponents()
        data = vtk_to_numpy(vtkArray).reshape(height, width, components)

        # # Examine geometry interactively is specified
        # # Create a renderwindowinteractor
        # renWin.OffScreenRenderingOn()
        # iren = vtk.vtkRenderWindowInteractor()
        # iren.SetRenderWindow(renWin)

        # # Enable user interface interactor
        # iren.Initialize()
        # ren.ResetCamera()
        # renWin.Render()
        # iren.Start()
        
        # Export image if desired
        if save:
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(self.imageFilename.as_posix())
            writer.SetInputConnection(imageFilter.GetOutputPort())
            writer.Write()

        
        return data
    
    def plot(self):
        
        # Based on 'http://people.sc.fsu.edu/~jburkardt/data/ply/beethoven.ply'
        
        
        # Open stl file and process relevant geometric properties
        filename = self.geometryFilename
        mesh = trimesh.load(filename.as_posix())
        vertices = mesh.vertices
        triangles = mesh.faces
        # triangles =  mesh_data.cells['triangle']
        # triangles =  mesh_data.cells
        x, y, z = vertices.T
        I, J, K = triangles.T

        pl_mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']

        # Plot surfaces
        pl_mesh = go.Mesh3d(x=x,
                            y=y,
                            z=z,
                            colorscale=pl_mygrey, 
                            intensity= z,
                            flatshading=True,
                            i=I,
                            j=J,
                            k=K,
                            name='Unit cell',
                            showscale=False,
                            )

        pl_mesh.update(cmin=-7,# atrick to get a nice plot (z.min()=-3.31909)
                       lighting=dict(ambient=0.2,
                                     diffuse=1,
                                     fresnel=0.1,
                                     specular=0.8,
                                     roughness=0.05,
                                     facenormalsepsilon=1e-15,
                                     vertexnormalsepsilon=1e-15),
                       lightposition=dict(x=100,
                                          y=200,
                                          z=0
                                         )
                      )

        layout = go.Layout(
                     title="Unit cell",
                     font=dict(size=16, color='white'),
                     width=600,
                     height=600,
                     scene_xaxis_visible=False,
                     scene_yaxis_visible=False,
                     scene_zaxis_visible=False,
                     scene=dict(aspectmode='data'),
                     paper_bgcolor='rgb(50,50,50)',

                )

        fig = go.Figure(data=pl_mesh, layout=layout)
        
        return fig
