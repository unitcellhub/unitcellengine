import json
import os
import subprocess
import tempfile
from numpy import interp
#import time
import datetime
import numpy as np
import plotly.graph_objects as go
import meshio  
from pathlib import Path
import tables

from tables.exceptions import NoSuchNodeError
import unitcellengine.geometry.sdf as geometry
from unitcellengine.geometry import (DEFAULT_THICKNESS, DEFAULT_ELEMENT_SIZE, 
                               DEFINITION_FILENAME, PROPERTIES_FILENAME)
import unitcellengine.mesh.internal as mesh
import unitcellengine.analysis.homogenization as homog
import logging
import logging.handlers as handlers
from unitcellengine.utilities import cachedProperty, timing
import shutil
import json
from tables import (IsDescription, Float64Col, StringCol, UInt8Col, 
                    open_file, Int16Col)
from datetime import datetime
# import unitcell.mesh.cubit as cubit
import multiprocessing
from multiprocessing.pool import ThreadPool
from PIL import Image

# Create logger
# logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Define HDF class for unit cell designs
IMG_WIDTH = 200
IMG_HEIGHT = 200
class HDFDesign(IsDescription):
    unitcell = StringCol(60)
    form = StringCol(20)
    length = Float64Col()
    width = Float64Col()
    height = Float64Col()
    thickness = Float64Col()
    radius = Float64Col()
    relativeDensity = Float64Col()
    relativeSurfaceArea = Float64Col()
    youngsModulus = Float64Col()
    poissonRatio = Float64Col()
    homogenizedStiffness = Float64Col(shape=(6,6))
    homogenizedCompliance = Float64Col(shape=(6,6))
    anisotropyIndex = Float64Col()
    vonMisesWorst11 = Float64Col()
    vonMisesWorst22 = Float64Col()
    vonMisesWorst33 = Float64Col()
    vonMisesWorst = Float64Col()
    vonMisesWorstDir = Float64Col(shape=(6))
    Emax = Float64Col()
    EmaxDirection = Float64Col(shape=(3))
    Emin = Float64Col()
    EminDirection  = Float64Col(shape=(3))
    Kmax = Float64Col()
    KmaxDirection  = Float64Col(shape=(3))
    Kmin = Float64Col()
    KminDirection  = Float64Col(shape=(3))
    Gmax = Float64Col()
    GmaxDirection  = Float64Col(shape=(3))
    GmaxNormal = Float64Col(shape=(3))
    Gmin = Float64Col()
    GminDirection  = Float64Col(shape=(3))
    GminNormal = Float64Col(shape=(3))
    numax = Float64Col()
    numaxDirection  = Float64Col(shape=(3))
    numaxNormal = Float64Col(shape=(3))
    numin = Float64Col()
    numinDirection  = Float64Col(shape=(3))
    numinNormal = Float64Col(shape=(3))

    conductance = Float64Col()
    homogenizedConductance = Float64Col(shape=(3,3))

    date = Int16Col(shape=(6))
    image = UInt8Col(shape=(IMG_WIDTH, IMG_HEIGHT, 4))

# def readDatabase(filename):
#     """ Read in H5 database file """

#     with open_file(filename, 'r') as h5file:
#         for row in h5file.root.database.iterrows():
#             print(row)

# homog = dict(E=result['E'], nu=result['nu'],
#                      stiffness=np.numpy(result['CH']),
#                      compliance=np.numpy(result['SH']),
#                      anisotropyIndex=result['anisotropyIndex'],
#                      vonMisesWorst11=result['amplification']['vm11'],
#                      vonMisesWorst22=result['amplification']['vm22'],
#                      vonMisesWorst33=result['amplification']['vm33'],
#                      vonMisesWorst=result['amplification']['vmWorst'][0],
#                      vonMisesWorstDir=result['amplification']['vmWorst'][1])


class UnitcellDesign(object):

    def __init__(self, unitcell, length, width, height, 
                 thickness=DEFAULT_THICKNESS, radius=0,
                 elementSize=DEFAULT_ELEMENT_SIZE,
                 directory="Database", form=None):
        """ Initial a lattice unitcell design 
        
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
        elementSize: float > 0 (default = depends on unitcell)
            Defines the normalized element size (with respect to the
            unit cell thickness parameter) used to calculate geometric
            properties. 
        directory: str of Path (default="Database")
            Defines the base database output folder where results will
            be stored.
        form: None, "graph", or "walled tpms" (Default=None)
            Defines the unitcell form. If None, the form is
            automatically determined based on the *unitcell* name.
        """
        # Specify the database directory
        self.directory = Path(directory)



        # Create the baseline unitcell geometry
        self.geometry = geometry.SDFGeometry(unitcell, length, width, 
                                          height, thickness=thickness, radius=radius,
                                          elementSize=elementSize, 
                                          form=form)
        self.geometry.directory = self.outputPath

        # # Create logger for unitcell instance. Write all output to log
        # # file.
        # logger = logging.getLogger(__name__)
        # logger.setLevel(logging.WARNING)
        # handler = handlers.RotatingFileHandler(self.outputPath / Path("run.log"),
        #                                        maxBytes=1e6, backupCount=10)
        # formatter = logging.Formatter(logging.BASIC_FORMAT)
        # handler.setFormatter(formatter)
        # handler.setLevel(logging.DEBUG)
        # logger.addHandler(handler)

        # handler2 = logging.StreamHandler()
        # handler2.setFormatter(formatter)
        # handler2.setLevel(logging.INFO)
        # logger.addHandler(handler2)
        # self.logger = logger


        self._homogenizationElastic = None
        self._homogenizationConductance = None

        self.cache = True


    
    def __repr__(self):
        geometry = self.geometry
        return f"{self.__class__.__name__}({geometry.unitcell}, " +\
               f"L={geometry.length:.3f}, " +\
               f"W={geometry.width:.3f}, H={geometry.height:.3f}, " +\
               f"T={geometry.thickness:.3f})"
    
    @property
    def databaseFilename(self):
        """ Export database filename """
        return self.directory / Path('database.h5')

    def generateGeometry(self, **kwargs):
        """ Generate the design geometry 
        
        Keywords
        --------
        reuse: boolean (default=False)
            If true and a geometry file already exists, don't recompute.
            Otherwise, compute the geometry.
        export: boolean (default=False)
            If true, export an STL representation of the geometry.

        Returns
        -------
        None
        """
        reuse = kwargs.pop("reuse", False)
        export = kwargs.pop("export", False)
        # if not reuse:
        self.geometry.run(reuse=reuse, export=export, **kwargs)
        # self.geometry.postprocess(reuse=reuse, engines=['blender'])
        return None
    
    def generateMesh(self, elementSize=0.2, reuse=False, **kwargs):
        """ Mesh the design geometry
        
        Keywords
        --------
        elementSize: float > 0 (default=0.2)
            Defines the element size relative to the thickness parameter
            for the design.  So, an *elementSize* of 0.1 corresponds to
            roughly 10 element through the thickness of the geometry.
        reuse: boolean (default=False)
            If true and a mesh file already exists, don't remesh.
            Otherwise, mesh the geometry.
        
        Returns
        -------
        None
        """

        # Check to see if the STL geometry exists. If not, generate it.
        if not self.geometry.processed:
            self.generateGeometry(reuse=False)
        
        # Generate the mesh (if it doesn't already exist)
        if reuse and self.meshFilename.exists():
            logger.info("Mesh reuse specified and mesh file "
                        f"{self.meshFilename} already exists. This mesh "
                        "is used rather than generating a new one. If "
                        "this is not the desired behavior, set 'reuse=False'")
            return
        else:
            # Remove any existing files
            if self.meshFilename.exists():
                logger.info(f"Mesh file {self.meshFilename} already exists. "
                            "Removing it prior to mesh creation.")
                self.meshFilename.unlink()

            # Create mesh
            L = self.geometry.length*self.geometry.DIMENSION
            W = self.geometry.width*self.geometry.DIMENSION
            H = self.geometry.height*self.geometry.DIMENSION
            T = self.geometry.thickness*self.geometry.DIMENSION
            mesh.mesh(self.geometry.sdf, elementSize*T, 
                      dimensions=[L, W, H], 
                      filename=self.meshFilename,
                      **kwargs)
    
    # def checkMesh(self, criteria):
    #     """ Check mesh shape quality 
        
    #     Arguments
    #     ---------
    #     criteria: 0 <= float < 1
    #         Defines the shape quality criteria to check against

    #     Returns
    #     -------
    #     Number of elements failing criteria check
    #     """

    #     # Calculate the mesh shape index
    #     quality = cubit.hexMeshQuality(self.meshFilename)

    #     # Check the quality against the given criteria
    #     check = quality < criteria

    #     return check.sum()

    # def homogenizeElastic(self, reuse=True, blocking=True):
    #     # Run homogenization simulations
    #     ps = self.homogenizationElastic.run(reuse=reuse, 
    #                                         blocking=blocking)
        
    #     # Process homogenization results

    
    @property
    def meshFilename(self):
        return self.outputPath / Path('unitcellMesh.npz')

    @property
    def homogenizationElastic(self):

        # Check to see if the homogenization object already exists
        if self._homogenizationElastic:
            return self._homogenizationElastic

        # Make sure the mesh files exists
        # if not self.meshFilename.exists():
        #     raise RuntimeError("Unit cell mesh has not been generated "
        #                        "yet. Run 'generateMesh' method and try"
        #                        "again.")
        
        # Homogenization object doesn't exist. Create it.
        obj = homog.InternalElasticHomogenization(self.meshFilename, 
                                                  E=1, nu=0.3
                                                  )
        self._homogenizationElastic = obj
        return obj
    
    @property
    def homogenizationConductance(self):

        # Check to see if the homogenization object already exists
        if self._homogenizationConductance:
            return self._homogenizationConductance

        # Make sure the mesh files exists
        # if not self.meshFilename.exists():
        #     raise RuntimeError("Unit cell mesh has not been generated "
        #                        "yet. Run 'generateMesh' method and try"
        #                        "again.")
        
        # Homogenization object doesn't exist. Create it.
        obj = homog.InternalConductanceHomogenization(self.meshFilename, 
                                                      k=1)
        self._homogenizationConductance = obj
        return obj

#     @property
#     def processed(self):
#         """ Has the design been processed? """
#         try:
#             if os.listdir(self.outputPath) > 4:
#                 return True
#             else:
#                 return False
#         except:
#             return False
    
    @cachedProperty
    def outputPath(self):
        # Define output directory based on unit cell parameters
        g = self.geometry
        folder = Path(f"L{g.length:.3f}_W{g.width:.3f}_H"
                      f"{g.height:.3f}_T{g.thickness:.4f}".replace(".", "_"))
        outputPath = self.directory / Path(g._cellForm) / Path(g.unitcell) / folder
                                  
        # Make sure folder exists
        outputPath.mkdir(parents=True, exist_ok=True)
        
        return outputPath
    
    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, value):
        # Check to see if directory exists
        value = Path(value)
        if not value.is_absolute():
            value = Path.cwd() / value
        
        value.mkdir(exist_ok=True)

        self._directory = value
    
    # @cachedProperty
    # def properties(self):
    #     """ Defined/calculated design properties """

    #     properties = {}

    #     # Pull in the geometry data
    #     properties.update(self.geometry.properties)
        
    #     # Pull in the homogeniation data
    #     result = self.homogenizationElastic.result
    #     CH = np.numpy(result['CH'])
    #     homog = dict(E=result['E'], nu=result['nu'],
    #                  stiffness=np.numpy(result['CH']),
    #                  compliance=np.numpy(result['SH']),
    #                  anisotropyIndex=result['anisotropyIndex'],
    #                  vonMisesWorst11=result['amplification']['vm11'],
    #                  vonMisesWorst22=result['amplification']['vm22'],
    #                  vonMisesWorst33=result['amplification']['vm33'],
    #                  vonMisesWorst=result['amplification']['vmWorst'][0],
    #                  vonMisesWorstDir=result['amplification']['vmWorst'][1])
    #     properties.update(homog)
    #    
    #     return properties
    
    def export(self):
        """ Export unit cell design properties to database """

        def _export(row):
            # Export geometry data
            for d in ['unitcell', 'length', 'width', 'height',
                      'thickness', 'radius', 'form', 'relativeDensity',
                      'relativeSurfaceArea']:
                      row[d] = getattr(self.geometry, d)
            
            if not self.geometry.imageFilename.exists():
                self.geometry.exportImage(save=True)
            img = Image.open(self.geometry.imageFilename)
            row['image'] = np.asarray(img.resize((IMG_HEIGHT, IMG_WIDTH)))

            # Export elastic homogenization results
            result = self.homogenizationElastic.result
            row['youngsModulus'] = result["E"]
            row['poissonRatio'] = result["nu"]
            row['homogenizedStiffness'] = np.array(result["CH"])
            row['homogenizedCompliance'] = np.array(result["SH"])
            row['anisotropyIndex'] = result['anisotropyIndex']
            row['vonMisesWorst11'] = result['amplification']['vm11']
            row['vonMisesWorst22'] = result['amplification']['vm22']
            row['vonMisesWorst33'] = result['amplification']['vm33']
            row['vonMisesWorst'] = result['amplification']['vmWorst'][0]
            row['vonMisesWorstDir'] = np.array(result['amplification']['vmWorst'][1])
            row['date'] = datetime.now().utctimetuple()[:6]
            for econst in ['E', 'K', 'G', 'nu']:
                for ext in ['min', 'max']:
                    n = econst+ext
                    subresult = result['engineeringConstants']
                    row[n] = subresult[n]['value']
                    row[n+'Direction'] = subresult[n]['d']
                    if econst in ['G', 'nu']:
                        row[n+'Normal'] = subresult[n]['n']

            # Export conductance homogenization results
            result = self.homogenizationConductance.result
            row['conductance'] = result["k"]
            row['homogenizedConductance'] = np.array(result["CH"])
        

        # Create a copy of the current database in case
        # anything goes wrong
        filename = self.databaseFilename
        if filename.exists():
            shutil.copy(filename,
                        filename.with_name(filename.stem+"_backup.h5"))

        with open_file(self.databaseFilename, mode='a', 
                       title="Lattice unitcell database") as h5file:
            # Check to see if the database exists yet. If not, create it.
            # try:
            #     group = h5file.root.database
            # except:
            #     group = h5file.create_group("/", "database", "Database")
            
            try:
                table = h5file.root.design

                # If table exists, make sure it has the correct format
                description = table.description
                ref = description.__dict__["_v_names"]
                diff = list(set(HDFDesign.columns.keys()) - set(ref))

                if diff:
                    text = (f"Database {self.databaseFilename} doesn't "
                            "seem to be the right format. Seems to be "
                            f"missing columns {diff}. Adding the "
                            "missing columns.")
                    logger.debug(text)
                    # raise IOError(text)

                    # Current table is inconsistent with the desired 
                    # format. Copy the existing table over into a new 
                    # one with the right format.
                    # See https://github.com/PyTables/PyTables/blob/master/examples/add-column.py
                    # for reference
                    tmp = h5file.create_table(h5file.root, "tmp", HDFDesign, 
                                              "Unit cell design",
                                              filters=tables.Filters(1))
                    # Copy the user attributes
                    table.attrs._f_copy(tmp)
                    
                    # Fill the rows of new table with default values
                    for i in range(table.nrows):
                        tmp.row.append()
                    
                    # Flush the rows to disk
                    tmp.flush()

                    # Copy the columns of source table to destination
                    for col in table.description._v_colobjects:
                        getattr(tmp.cols, col)[:] = getattr(table.cols, col)[:]
                    
                    # Remove the original table
                    table.remove()

                    # Move the new table into the original tables location
                    tmp.move('/', 'design')
                    table = tmp
                else:
                    flag = True
                    try:
                        for column, value in HDFDesign.columns.items():
                            if not all([getattr(getattr(description, column), k) == getattr(value, k) 
                                    for k in ['kind', 'size', 'shape']]):
                                flag = False
                                break
                    except AttributeError:
                        flag = False
                    
                    if not flag:
                        text = (f"Database {self.databaseFilename} "
                                "appears to have the correct columns "
                                "but they don't appear to be formatted "
                                "correctly.")
                        logger.error(text)
                        raise IOError(text)


                # Note: this is untested but could be used to update the existing
                # database as needed to accommodate the new format.
                # if not flag:
                #     # Current table is inconsistent with the desired 
                #     # format. Copy the existing table over into a new 
                #     # one with the right format.
                #     # See https://github.com/PyTables/PyTables/blob/master/examples/add-column.py
                #     # for reference

                #     tmp = h5file.create_table(h5file.root, "tmp", HDFDesign, 
                #                               "Unit cell design")

                #     # Fill the rows of new table with default values
                #     for i in range(table.nrows):
                #         tmp.row.append()
                    
                #     # Flush the rows to disk
                #     tmp.flush()

                #     # Copy the columns of source table to destination
                #     for col in table.description._v_colobjects:
                #         getattr(tmp.cols, col)[:] = getattr(table.cols, col)[:]
                    
                #     # Remove the original table
                #     table.remove()

                #     # Move the new table into the original tables location
                #     tmp.move('/tmp', 'design')

            except NoSuchNodeError:
                table = h5file.create_table(h5file.root, "design", HDFDesign, 
                                           "Unit cell design")

            # Check to see if the design already exists. If so, update it.
            # If not, add the design to the database
            def isclose(name, value, precision=5):
                return f"({name} < {value+10**(-precision)}) & " +\
                       f"({name} > {value-10**(-precision)})"
            condition = " & ".join([isclose('length', self.geometry.length),
                                    isclose('width', self.geometry.width),
                                    isclose('height', self.geometry.height),
                                    isclose('thickness', self.geometry.thickness)])
            condition += f' & (unitcell == b"{self.geometry.unitcell}")'
            condition += f' & (form == b"{self.geometry._cellForm}")'
            counter = 0
            for row in table.where(condition):
                _export(row)
                row.update()
                logger.debug(f"Updated {self} in the {self.databaseFilename} "
                             "database.")
                counter += 1
            
            if counter == 1:
                return
            elif counter == 0:
                # Design doesn't exist yet. Add in a new row to the
                # table. 
                row = table.row
                _export(row)
                row.append()
                logger.debug(f"Added {self} to the {self.databaseFilename} "
                             "database.")
            else:
                raise RuntimeError("Too many rows were matched in the database. "
                                   "There appear to be duplicates that should "
                                   "be cleaned up or the database is corrupt.")
            # Flush the results to write them to file
            table.flush()

    
    # @cachedProperty
    # def properties(self):
    #     """ Unit cell properties """

    #     # Create baseline dictionary
    #     props = {}

    #     # Store geometric properties
    #     geom = self.geometry
    #     subprops = props['geometry'] = {}
    #     subprops['unitcell'] = geom.unitcell
    #     subprops['length'] = geom.length
    #     subprops['width'] = geom.width
    #     subprops['height'] = geom.height
    #     subprops['thickness'] = geom.thickness
    #     subprops['relativeDensity'] = geom.relativeDensity
    #     subprops['relativeArea'] = geom.properties['Relative surface area']
    #     subprops['image'] = geom.exportImage()

    #     # Store homogenization properties
    #     homog = self.homogenizationElastic
    #     subprops = props['mechanical'] = {}
    #     CH = homog.CH
    #     subprops['stiffness'] = {}


    # # @property
    # @cachedProperty
    # def properties(self):
    #     properties = {}
    #     try:
    #         # Read unit cell propreties data
    #         with (self.outputPath / Path("unitcellProperties.txt")).open('r') as f:
    #             data = f.read()

    #         # Parse the unit cell properties data
    #         for line in data.split(r'\r'):
    #             split = line.split(': ')
    #             properties[split[0]] = float(split[1])

    #         # Open the homogenization data
    #         data = np.genfromtxt(os.path.join(self.outputPath, 
    #                                           "homogenizedProperties.txt"))
    #         properties['Homogenized stiffness'] = data

    #     except FileNotFoundError:
    #         raise ValueError(f"Design {self} not run or failed during run.")

    #     return properties
    
    # def delete(self):
    #     """ Delete objected and all corresponding files """
    #     logger.info(f"Deleting {self.outputPath} and all subfolders.")
    #     shutil.rmtree(self.outputPath)
    #     assert not self.outputPath.exists()

    def plot(self):
        """ Plot unit cell geometry """
        
        return self.geometry.plot()


class UnitcellDesigns(object):
    """ Batch of lattice designs and corresponding interpretation """

    @timing(logger=logger)
    def __init__(self, unitcell, directory='Database', form=None):
        """ Load set of existing designs """
        
        # Define base design properties
        self._unitcell = unitcell
        self._directory = directory

        # Parse the cell form definition
        if form not in ['graph', 'walledtpms']:
            raise ValueError("Form must be either 'graph' or 'walledtpms', "
                             f"not {form}.")
        self._cellForm = form

        # Define search path for pre-existing designs
        search = Path(directory) / Path(form) / Path(unitcell)
        logger.debug(f'Searching {search} for unit cell designs.')

        # # Determine what constructor to use when reading in design information
        # if unitcell in _GRAPH_UNITCELLS:
        #     constructor = GraphLatticeDesign
        # elif unitcell in _WALLED_TPMS_UNITCELLS:
        #     constructor = WalledTPMSLatticeDesign
        # else:
        #     raise ValueError(f"Unitcell {unitcell} is not valid. Must be "
        #                      f"in the list ({', '.join(_GRAPH_UNITCELLS)}) "
        #                      f"or ({', '.join(_WALLED_TPMS_UNITCELLS)})")

        # Loop through each design folder and load the corresponding
        # data
        designs = []
        try:
            with ThreadPool() as p:
                for design in p.imap(self._checkFolder, search.iterdir()):
                    if design:
                        designs.append(design)

        except FileNotFoundError:
            pass

        self.designs = designs

    def _checkFolder(self, folder):
        """ Check a given folder for an existing design 
        
        Arguments
        ---------
        folder: str or Path
            Folder to check
        
        Returns
        -------
        UnitcellDesign object or None
        
        """

        # Make sure folder is a Path object
        folder = Path(folder)

        try:
            # Open up the input json file and pull out the geometry information
            with (folder / DEFINITION_FILENAME).open('r') as f:
                definition = json.load(f)
            definition["directory"] = self._directory
            
            # Read in design                  
            design = UnitcellDesign(**definition)
            logger.debug(f'Found design {design} in {folder}. Adding it to the list.')
            return design
        except FileNotFoundError:
            logger.debug(f'No data found in {folder}. Skipping.')
        except json.JSONDecodeError:
            logger.debug('Issue reading json file '
                        f'{folder / DEFINITION_FILENAME}. Skipping.')
        return None


    def addDesign(self, length, width, height, thickness=DEFAULT_THICKNESS, 
                  elementSize=DEFAULT_ELEMENT_SIZE, radius=0.):
        """ Add design to design set 
        
        Arguments
        ---------
        length: float > 0
            Non dimensional unit cell length.
        width: float > 0
            Non dimensional unit cell width.
        height: float > 0
            Non dimensional unit cell height.
        
        Keywords
        --------
        thickness: float > 0 or None [Default]
            Non dimensional unit cell thickness. If none, use the
            default values.
        elementSize: float > 0 [Default]
            Relative geometry surface mesh element size.
        radius: float > 0 [Default = 0.]
            Relative joint radius size
        
        Returns
        -------
        UnitCell design or False if the design already exists.
        """

        # Check to see if design already exists. If it does, raise a
        # warning and skip it.
        parameters = np.array([length, width, height, thickness, radius])
        try:
            for existing in self.geometryArray:
                if np.allclose(parameters, existing, atol=1e-3):
                    logger.warning(f"Design with length={length}, width={width}, "
                                f"height={height}, thickness={thickness}, "
                                "and radius={radius}"
                                "already exists. Design was not added to "
                                "design space.")
                    return False
        except AttributeError:
            # This handls the case in which there are no existing
            # designs 
            pass
        

        # Create design
        design = UnitcellDesign(self._unitcell, 
                                length=length,
                                width=width, 
                                height=height, 
                                thickness=thickness,
                                elementSize=elementSize,
                                radius=radius,
                                directory=self._directory,
                                form=self._cellForm)
        
        # Add design to design set
        logger.debug(f'Created/loading {design}. Adding to the list.')
        self.designs.append(design)

        return design

    def structuredSampling(self, samplesL, samplesW, samplesH, 
                           samplesT=[DEFAULT_THICKNESS], 
                           elementSize=DEFAULT_ELEMENT_SIZE,
                           radius=0.):
        """ Sample design space on a structured grid 
        
        Arguments
        ---------
        sampleL, sampleW, sampleZ: array-likes
            Defines grid points to sample in the length, width, and
            height, respectively. 
        
        Keywords
        --------
        sampleT: array like or None [Default]
            Thickness grid to sample. If None, use default thickness. If
            specified, include in structured grid.
        elementSize: float > 0 [Default]
            Normalized element size.
        
        Returns
        -------
        list of UnitcellDesign objects
        """

        # Mesh the structured grid
        grid = np.meshgrid(samplesL, samplesW, samplesH, samplesT)
        points = np.array([g.flatten() for g in grid]).T

        # Create designs
        designs = []
        for L, W, H, T in points:
            # Create design
            design = self.addDesign(L, W, H, thickness=T,
                                    elementSize=elementSize, 
                                    radius=radius)
            
            # Append to list
            designs.append(design)
        
        return designs



    @cachedProperty
    def N(self):
        """ Number of designs """
        return len(self.designs)

    def geometryList(self, prop):
        return [design.geometry.__getattribute__(prop) for design in self.designs]
    
    @cachedProperty
    def geometryArray(self):
        return np.array([[design.geometry.length, design.geometry.width, 
                         design.geometry.height, design.geometry.thickness, 
                         design.geometry.radius] for design in self.designs])
    
    def propertyList(self, prop):
        # out = []
        # for design in self.designs:
        #     try:
        #         out.append()
        return [design.properties[prop] for design in self.designs]
    
    @cachedProperty
    def thicknesses(self):
        return self.geometryList('thickness')
    
    @cachedProperty
    def lengths(self):
        return self.geometryList('length')
    
    @cachedProperty
    def widths(self):
        return self.geometryList('width')
    
    @cachedProperty
    def heights(self):
        return self.geometryList('height')
    
    @cachedProperty
    def relativeDensities(self):
        return self.geometryList('relativeDensity')

    @cachedProperty
    def relativeSurfaceAreas(self):
        return self.geometryList('relativeSurfaceArea')
    
    @cachedProperty
    def homogenizedStiffnesses(self):
        return self.propertyList('Homogenized stiffness')
    
    @cachedProperty
    def properties(self):
        """ Unit cell defined/calculated propreties"""

        properties = []
        for design in self.designs:
            properties.append(design.properties)
        
        return properties
    
if __name__ == "__main__":

    # design = UnitcellDesign('Column', 1, 1, 1, thickness=0.15,
    #                         directory=Path("Database"))
    # design.generateMesh(elementSize=0.5, reuse=False)
    logging.basicConfig(level=logging.DEBUG)
    # database = Path(r'E:\Lattice\Database')
    # design = UnitcellDesign('SplitP' 1, 5, 1, thickness=0.30,
    #                         directory=database, form='walledtpms')
    # # design.geometry.elementSize = 0.15
    # # design.geometry.run(reuse=False, headless=False)
    # # design.generateGeometry(reuse=False)
    # design.generateMesh(0.34, reuse=False, nprocessors=4)
    # design.homogenizationElastic.run(nprocessors=8, reuse=False)
    # design.homogenizationElastic.process(check=False)
    # # design.generateGeometry(reuse=False)
    # # print(design)
    # designs = []
    # # design = UnitcellDesign('Schwarz', 1.5, 3, 4.25, elementSize=0.15)
    # failed = []
    # reuse = True
    # sampling = np.array([1, 1.25, 1.5, 2, 3, 4, 5])
    # # for form, unitcells in zip(['walledtpms', 'graph'],
    # #                            [ntop._WALLED_TPMS_UNITCELLS,
    # #                             ntop._GRAPH_UNITCELLS]):
    # for form, unitcells in zip(['graph'],
    #                            [ntop._GRAPH_UNITCELLS]):
    #     for unitcell in unitcells:
    #         print(f"Creating {unitcell} unit cell geometry")
    #         subdesigns = UnitcellDesigns(unitcell, database, form=form)
    #         samples = subdesigns.structuredSampling(sampling, sampling, 
    #                                                 sampling, samplesT=[0.15],
    #                                                 elementSize=0.2)
    #         designs += samples
    
    # # Generate geometry for each sample
    # bar = progressbar.ProgressBar(maxval=len(designs), \
    #             widgets=[progressbar.Bar('=', '[', ']'), ' ', 
    #                      progressbar.Percentage()])
    # bar.start()
    # for i, design in enumerate(designs):
    #     bar.update(i+1)
    #     try:
    #         design.generateGeometry(headless=True, reuse=reuse)
    #     except:
    #         design.elementSize = 0.15
    #         try:
    #             design.generateGeometry(headless=True, reuse=reuse)
    #         except:
    #             failed.append(design)
    # bar.finish()
            # try:
            #     design.generateGeometry(headless=True, reuse=False)
            # except:
            #     design.elementSize = 0.15
            #     try:
            #         design.generateGeometry(headless=True, reuse=False)
            #     except:
            #         failed.append(unitcell)
    # cases = dict()

    # cases = []

    # failed = []
    # for case in cases:
    #     try:
    #         design = UnitcellDesign(*case, thickness=0.3, 
    #                                 elementSize=0.15, directory=Path(r"F:\Lattice\Database"),
    #                                 form='graph')
    #         # design.homogenizationElastic.run(reuse=False, nprocessors=8)
    #         # design.homogenizationElastic.plotG()
    #         design.homogenizationElastic.process(check=False, reuse=False)
    #     except AssertionError as ex:
    #         print(f"Design {design} failed with {ex}")
    #         failed.append(design)

    # design = UnitcellDesign("Re-entrant honeycomb", 1.25, 4, 4, thickness=0.3, 
    #                         elementSize=0.025, directory=Path(r"F:\Lattice\Database"),
    #                         form='graph')

    # design.homogenizationElastic.process(check=False, reuse=False)
    # import re
    # regex = re.compile(r"\((.*?), L=(.*?), W=(.*?), H=(.*?), T=(.*?)\)")
    # designs = []
    # with open("failed.txt", 'r') as f:
    #     for line in f:
    #         # if "mesh file" in line:
    #         match = regex.search(line)
    #         results = [arg if i ==0 else float(arg) for i, arg in enumerate(match.groups())]

    #         designs.append(results)

    # design = UnitcellDesign(*designs[2], 
    #                         elementSize=0.025, 
    #                         directory=Path(r"F:\Lattice\Database"),
    #                         form="graph")
    # design.generateMesh(elementSize=0.3, reuse=False, nprocessors=12)

    # designs = UnitcellDesigns("Body centered cubic", form="graph")
    # designs.structuredSampling([1.], [1.], [1.], [0.2], radius=0.2)

    # design.homogenizationElastic.run(nprocessors=12, reuse=False, cases="all")
    # design.homogenizationElastic.process(check=False, reuse=False)
    # designs = [ ]

    design = UnitcellDesign("Diamond", 4, 4, 3.75, thickness=0.26, 
                            radius=0.7,
                            elementSize=0.2,
                            form='graph', 
                            directory=Path(r"Database/"))
    design.generateGeometry(reuse=True)
    design.generateMesh(reuse=True)
    design.homogenizationElastic.run(reuse=True)

    design.homogenizationElastic.process(reuse=False, check=False)
    # stress = h.localVMStress([1, 0, 0, 0, 0, 0])
    # stress = h.localVMStress(np.array([[1, 1, 1, 1, 1, 1]]))
    # stress = h.localVMStress(np.array([[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]).T)

    for i in range(3):
        print(f"E_{i+1}: {design.homogenizationElastic.Ei(np.eye(3)[i, :])}")

    for i, j in [[0, 1], [1, 2], [0, 2]]:
        print(f"G_{i},{j}: {design.homogenizationElastic.Gij(np.eye(3)[i, :], np.eye(3)[j, :])}")
        print(f"nu_{i},{j}: {design.homogenizationElastic.nuij(np.eye(3)[i, :], np.eye(3)[j, :])}")

    # macroStresses = np.array([
    #         [-0.004049, -0.007476, -0.000172, -3.777e-05, 0.01209, 1.896e-05],
    #         [0.003043, -0.002334, 0.008725, -6.022e-05, 0.008914, 5.436e-05],
    #         [-0.003643, -0.004924, -0.002088, 1.237e-05, 0.01294, -1.071e-05],
    #         [-0.001522, -0.002338, -0.0006459, 2.856e-05, 0.01057, -6.88e-06],
    #         [-0.000468, -0.0008556, -0.0001219, 1.987e-05, 0.00953, 6.911e-06],
    #         [0.005936, 0.002729, 0.01105, 9.042e-06, 0.006727, 5.15e-05],
    #         [0.004901, 0.003373, 0.009675, 0.0001992, 0.005162, -0.0001258],
    #         [0.005073, 0.003433, 0.0102, 9.245e-05, 0.004699, -0.0001412],
    #         [0.01757, 0.01458, 0.02172, -0.0002137, -0.003001, 0.0001582],
    #         [0.01031, 0.004902, 0.01762, -0.0001443, 0.001094, -2.816e-05],
    #         [0.006801, 0.000842, 0.01421, 7.226e-05, 0.002584, -6.572e-05],
    #         [0.006073, 0.00283, 0.01209, -8.076e-05, 0.003797, 2.519e-05],
    #         [0.01294, 0.01294, 0.01392, 1.288e-05, -0.0002015, 8.93e-07],
    #         [0.01308, 0.01312, 0.01416, -5.256e-05, -0.0008898, 1.206e-05],
    #         [0.01695, 0.01607, 0.01937, 1.072e-05, 0.004743, -6.784e-05],
    #         [0.01626, 0.01328, 0.02098, 0.0001025, 0.003037, -1.428e-05],
    #         [0.01289, 0.01285, 0.01399, 6.534e-05, 0.000562, -2.703e-05],
    #         [0.01321, 0.01317, 0.01401, 6.693e-05, 0.00141, -0.000113],
    #         [0.01368, 0.01347, 0.01547, 6.069e-05, 0.002361, -0.0001595],
    #         [0.01365, 0.01366, 0.01461, -5.259e-05, -0.001544, -2.602e-05],
    #         [0.01462, 0.0145, 0.01607, -9.973e-05, -0.002581, -5.39e-05],
    #         [0.01727, 0.0165, 0.01906, -7.641e-05, -0.004462, 0.0001003],
    #         [0.007142, 0.001177, 0.01423, -7.029e-05, -0.003331, -0.000106],
    #         [0.006546, 0.0006463, 0.01354, 0.0002472, -0.004064, 7.574e-05],
    #         [0.007033, 0.0009924, 0.01455, -4.217e-05, -0.002701, -0.0002137],
    #         [0.009217, 0.003668, 0.01677, 7.126e-05, -0.00148, -0.0001881],
    #     ]).T
        macroStresses = np.array([
            [-9.144708, -7.480976, -5.632445, 0.043773, -0.292898, -5.782212],
            [-6.21604, -7.600909, -4.859807, -0.582898, 5.150481, 1.776366],
            [-4.469099, -5.546664, -3.343966, 0.400487, -5.015183, 1.950473],
            [-3.845384, -4.859985, -2.713056, 0.158518, -5.319328, 0.307216],
            [-4.092361, -5.007422, -3.049485, 0.715253, -4.832668, 2.137458],
            [-6.597293, -8.084064, -5.121946, -0.49077, 4.883836, 1.288616],
            [-5.768534, -4.594352, -3.324933, 0.266191, 0.849721, -4.928317],
            [-6.154285, -7.007194, -4.46169, 1.605717, 3.7847, -2.697588],
            [-5.356611, -6.789192, -3.879073, 0.479202, -4.78775, 0.745534],
            [-4.389032, -5.509865, -3.234537, -0.260542, -4.902423, -0.183011],
            [-3.142525, -3.591791, -0.645919, 2.390073, 3.171208, -2.670439],
            [-2.87645, -5.042558, -0.453313, 1.228176, -4.089492, 1.337862],
            [-4.250622, -3.264941, -2.271401, -0.321806, 1.374993, 4.658091],
            [-4.320498, -3.729004, -2.454876, 1.243991, 2.735896, -3.799251],
            [-3.36912, -4.278934, -2.306052, -0.190184, 4.822659, 0.047743],
            [-3.234969, -4.136112, -2.193104, 0.014392, 4.816833, -0.062786],
            [-3.159634, -3.840962, -2.459981, -0.180831, -4.702832, -1.200253],
            [-4.06183, -3.301246, -2.456786, -0.098698, -0.233017, 4.830405],
            [-4.296551, -3.374718, -2.352274, -0.568571, 1.610364, 4.481672],
            [-2.993019, -3.59187, -1.994528, -0.917755, 3.88334, 2.696839],
            [-4.764664, -3.724268, -2.670811, -0.296151, -1.120776, -4.617053],
            [-4.457998, -3.496113, -2.474715, -0.170194, -0.562052, -4.721012],
            [-2.200602, -4.290194, 0.144267, -1.290943, -3.737141, -1.700103],
            [-3.864128, -3.095925, -2.304784, 0.08263, 0.329932, 4.763511],
            [-2.547852, -3.271763, -1.985985, -0.101615, -4.688114, -0.916119],
            [-4.37086, -3.504629, -2.549349, -0.13441, 0.26331, -4.715104],
            [7.67867, 6.744054, 4.837538, -1.74182, -2.072088, 3.703717],
            [-3.520386, -4.397378, -2.657465, 0.258574, 4.546673, -1.19285],
            [-3.936463, -3.224335, -2.371322, 0.232345, -0.522011, 4.670245],
            [-2.61898, -2.587466, -0.259206, -2.031723, -2.900776, -2.890088],
            [-3.181032, -3.976086, -2.281527, 0.361358, 4.425631, -1.476718],
            [-3.842787, -2.978638, -2.149783, -0.327435, -1.545878, -4.402431],
            [-4.23118, -3.432799, -2.579662, -0.106725, 0.65565, 4.606059],
            [6.69534, 7.044417, 4.554191, 2.201031, 3.028442, 2.54986],
            [-2.981081, -2.977052, -2.050951, -0.874015, 3.337923, 3.179154],
            [6.55251, 7.223747, 4.602381, 2.113539, 3.276139, 2.277347],
            [-3.932265, -3.03503, -2.098445, 0.335983, 1.256389, -4.435214],
            [-3.493901, -2.824626, -1.999604, 0.654933, -2.253714, 4.007523],
            [-3.468698, -4.214978, -2.579735, 0.615874, 3.985313, -2.274363],
            [-3.665864, -2.933647, -2.051019, -0.488486, -1.806981, -4.215874],
            [-2.551495, -2.79189, -1.795588, -0.753894, 3.511649, 2.95842],
            [7.27636, 6.691348, 4.666688, 1.911211, -2.352319, -3.277492],
            [-4.326319, -3.446477, -2.406932, 0.626045, 1.524858, -4.250546],
            [6.562133, 7.007805, 4.48329, 2.207676, -3.020356, -2.420596],
            [-2.649575, -3.18616, -2.040079, -0.271618, -4.348048, -1.51199],
            [8.198083, 6.647258, 4.919256, 0.169252, 0.068377, 4.342501],
            [-2.204964, -2.264978, 0.20333, 2.080419, 2.802831, -2.714338],
            [7.201524, 6.472842, 4.509727, -1.982068, 2.223266, -3.273795],
            [-4.03532, -5.020386, -2.764527, -0.811962, 4.029058, 1.812011],
            [-3.663084, -2.999327, -1.977181, -0.878377, 2.320237, 3.812639],
            [7.455273, 6.526017, 4.616275, 1.654167, 1.820098, 3.626592],
            [-3.37866, -2.733968, -2.047682, -0.478614, 1.919544, 4.095972],
            [0.813728, -0.639088, 3.165779, 2.069094, 2.945261, -2.090448],
            [6.541196, 7.279082, 4.595637, 1.999928, -3.206845, -2.189786],
            [-3.682051, -3.043624, -2.07501, 0.797774, 2.255025, -3.79932],
            [-3.808689, -2.410617, -0.414511, 1.457071, 1.888096, -3.491984],
            [6.91239, 6.545671, 4.454282, -2.068945, 2.337149, -3.043918],
            [8.391017, 7.10834, 5.3403, -1.601622, -1.838784, 3.531341],
            [-2.632463, -3.075068, -1.748721, -0.841577, -3.56347, -2.625635],
            [-2.970387, -2.20094, -0.29104, -1.781406, 2.345503, 3.186363],
            [-1.736207, -3.875335, 0.63089, -1.098597, -3.544905, -1.349407],
            [-3.398942, -2.749868, -1.917588, -0.680775, -2.178768, -3.861389],
            [-3.066567, -2.426833, -1.823937, -0.160586, -1.07657, 4.360924],
            [-2.418399, -2.381789, 0.04718, 2.101851, -2.600109, 2.717334],
            [-2.726047, -1.997313, -0.046491, 1.842178, -2.331627, 3.124972],
            [0.568325, 0.297377, 4.570145, 3.81739, -0.171341, 0.253252],
            [-2.257747, -2.703287, -0.080196, -1.960725, -2.882279, -2.499269],
            [6.550859, 8.011265, 4.760458, -0.537105, 4.097683, -0.753294],
            [-2.517876, -2.019344, 0.18701, -1.940095, -2.515688, -2.84257],
            [-3.503756, -4.531136, -2.452177, 0.185898, 4.343471, -0.482335],
            [-2.514507, -2.182557, 0.123652, 2.039168, -2.481069, 2.78402],
            [-1.868033, -2.666768, -1.096424, -0.025693, -4.405796, 0.112791],
            [-2.68435, -1.979424, -0.028101, -1.90292, -2.372435, -2.97665],
            [6.671332, 8.181765, 4.973035, -0.431345, -4.117489, 0.496725],
            [8.180301, 6.602526, 4.894042, -0.314751, -0.371937, 4.121777],
            [-2.570811, -2.155091, 0.085074, 2.023434, 2.403277, -2.821819],
            [6.680352, 6.586627, 4.39615, 2.115612, 2.553862, 2.667785],
            [0.412771, -1.371097, 2.703526, 1.713332, -3.014519, 1.886066],
            [-1.076097, -1.718588, -0.39821, 0.356816, -4.013425, 1.735727],
            [-2.366287, -2.925418, -1.817264, -0.02189, -4.233537, 1.188576],
            [6.883495, 6.274486, 4.330504, -1.990804, -2.156641, 3.039811],
            [-1.784732, -2.194092, -0.610393, -1.063929, 3.317959, 2.579763],
            [-2.396692, -2.035242, 0.120614, -1.959555, 2.477086, 2.759984],
            [-2.819763, -3.353207, -2.193624, -0.529145, -3.715207, -2.224511],
            [7.760839, 6.312965, 4.670199, 0.968383, -0.930152, -3.891664],
            [-2.448483, -3.521857, -1.455182, 0.101426, -4.242853, 0.467225],
            [-1.886693, -2.651748, -1.154248, -0.180167, -4.160874, -1.178678],
            [-2.848829, -1.955562, -1.152682, 0.092929, 0.609286, -4.256312],
            [-2.876246, -2.30018, -1.734929, 0.26898, -1.778503, 3.953321],
            [7.82929, 6.336711, 4.586955, 0.73428, 0.730512, 3.930637],
            [7.043645, 6.167488, 4.274545, 1.687126, -2.001892, -3.205942],
            [6.476306, 7.315081, 4.630941, -1.67037, 3.300746, -1.873698],
            [6.904603, 6.439633, 4.491725, 2.008897, 2.273801, 2.869836],
            [7.387739, 6.18408, 4.401646, 1.272199, 1.41716, 3.632091],
            [-1.897644, -2.344134, 0.464403, -2.144513, 2.576621, 2.345343],
            [-1.64527, -2.40525, -0.898803, 0.032838, -4.293779, 0.003893],
            [-3.382116, -2.769782, -2.131921, 0.151118, -1.360625, 4.089357],
            [-2.569977, -1.902008, -1.292339, 0.082454, 0.238409, 4.299466],
            [7.818305, 6.300693, 4.511041, 0.386929, 0.362164, 3.986245],
            [-1.896801, -2.946269, 0.115856, -1.720922, 3.096662, 1.965135],
            [-2.138506, -3.448305, -0.40766, -1.297523, -3.45448, -1.695681],
            [-3.185032, -2.375152, -1.656515, -0.100032, 0.162917, -4.264459],
            [6.304693, 7.792628, 4.578336, -0.928201, 3.799491, -0.951684],
            [6.438009, 6.387245, 4.145599, -2.06698, 2.413869, -2.632884],
            [-1.991807, -2.7442, -1.303265, 0.042494, 4.260407, -0.255095],
            [7.586398, 6.208551, 4.425741, 1.057898, 1.125527, 3.715937],
            [0.631688, 1.063846, 4.940173, 3.597503, -0.337645, 0.0154],
            [-2.27099, -3.715161, -0.347396, 1.432867, 3.293141, -1.703229],
            [-1.681618, -2.200337, -1.197199, -0.111319, -4.169076, -0.981424],
            [-2.824349, -2.56577, -1.958894, -0.608013, -2.697863, -3.278748],
            [-1.985485, -2.352491, 0.20788, 1.920801, -2.701317, 2.384556],
            [-2.825841, -2.04864, -1.292553, -0.251492, 1.041503, 4.095601],
            [-2.719795, -1.652205, -0.544416, -0.402916, 0.985457, 4.018946],
            [6.36795, 7.324339, 4.527189, -1.599764, -3.320736, 1.67717],
            [-1.952521, -2.01494, 0.504034, -2.032697, 2.498999, 2.444874],
            [6.551118, 6.297054, 4.270001, 2.139411, -2.27407, -2.65862],
            [-1.230831, -2.11704, -0.282601, 0.566137, -3.800543, 1.658138],
            [-2.613268, -2.387527, -1.760825, 0.631418, -2.654406, 3.260339],
            [-2.621656, -1.839139, -1.049965, -0.027203, -0.2874, -4.186064],
            [6.551906, 6.31269, 4.306893, -1.969241, -2.314397, 2.718707],
            [-2.878873, -2.471267, -1.927424, -0.544793, -2.419045, -3.429485],
            [6.386079, 6.648977, 4.279768, -1.979676, 2.598261, -2.388985],
            [-2.8376, -2.036624, -1.285662, -0.344593, -1.138803, -4.003918],
            [-2.119857, -1.401097, -0.642127, -0.275812, 1.322564, 3.956975],
            [-2.516578, -2.501332, -1.800265, 0.663473, -2.877025, 3.021604],
            [-1.561922, -1.914394, -0.785482, -0.742226, 3.305833, 2.488551],
            [7.736248, 6.285892, 4.620985, -0.54282, -0.599459, 3.860273],
            [1.644104, 1.199143, 5.290367, -3.518951, -0.513482, -0.542686],
            [6.107921, 7.217905, 4.32743, -1.570929, 3.257734, -1.657903],
    ]).T*1e-3



    from unitcellengine.mesh.internal import convert2pyvista
    vmstress = design.homogenizationElastic.localVMStress(macroStresses)
    cellData = {f'vm{i}': vmstress[:, i] for i in range(vmstress.shape[1])}
    grid = convert2pyvista(design.meshFilename, cellData=cellData) 
    grid.save("vmtest.vtu")
    print(design.geometry.relativeDensity)
    print(design)
    # design = UnitcellDesign("Body centered cubic", 3, 3, 3, thickness=0.3, 
    #                         elementSize=0.2, radius=0.25,
    #                         directory=Path(r"F:\Lattice\Database"),
    #                         form='graph')
    
    # from unitcellengine.analysis.internal import internal2vtu

    # internal2vtu(design.meshFilename, design.homogenizationElastic._resultfile)

    # design.homogenizationConductance.process(reuse=False)
    # design.homogenizationElastic.process(reuse=False, check=False)
    # design.export()
    # import unitcell.mesh.internal as mesh
    # import unitcell.analysis.internal as analysis
    # design.generateGeometry(reuse=False)
    # design.generateMesh(elementSize=0.2, reuse=False)
    # mesh.internal2vtu(design.meshFilename)
    # design.homogenizationElastic.run(reuse=False)
    # # analysis.internal2vtu(design.meshFilename,
    # #                 design.homogenizationElastic._resultfile)
    # design.homogenizationElastic.process(reuse=False)
    # design.homogenizationConductance.run(reuse=False)
    # design.homogenizationConductance.process(reuse=False)
    # design.homogenizationElastic.process(check=True, reuse=False)
    # with open("failed_updates.txt", 'a') as f:
    #     for args in designs:
    #         form = "walledtpms" if args[0] in ["Gyroid", "Lidinoid"] else "graph"
    #         design = UnitcellDesign(*args,
    #                                 elementSize=0.025, directory=Path(r"F:\Lattice\Database"),
    #                                 form=form)
    #         # flag = False
    #         # for size in [0.1, 0.05, .025, 0.01]:
    #         #     try:
    #         #         design.geometry.elementSize = size
    #         #         design.generateGeometry(reuse=False, headless=True)
    #         #         # flag = True
    #         #         for size in [0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
    #         #             try:
    #         #                 design.generateMesh(size, reuse=False, nprocessors=12)
    #         #                 flag = True
    #         #                 break
    #         #             except:
    #         #                 continue
    #         #         if flag:
    #         #             break
    #         #     except:
    #         #         f.write(f"Design {design} failed: meshing failed\n")
    #         #         f.flush()
    #         #         continue
    #         # if not flag:
    #         #     f.write(f"Design {design} failed: meshing failed\n")
    #         #     f.flush()
    #         #     continue
    #         # try:
    #         #     design.generateMesh(0.20, reuse=False, nprocessors=12)
    #         # except RuntimeError as ex:
    #         #     f.write(f"Design {design} failed: Failed to mesh - {ex}\n")
    #         #     f.flush()
    #         #     break
    #         # if not design.meshFilename.exists():
    #         #     flag = False
    #         #     for size in [0.35, 0.25]:
    #         #         try:
    #         #             design.generateMesh(size, reuse=False, nprocessors=12)
    #         #             flag = True
    #         #             break
    #         #         except:
    #         #             continue
    #         #     if not flag:
    #         #         f.write(f"Design {design} failed: could mesh file\n")
    #         #         continue
    #         # check = design.homogenizationElastic.check()

    #         # print(design.homogenizationElastic.check())

    #         for size in [0.2, 0.15, 0.1, 0.05, .025]:
    #             try:
    #                 design.geometry.elementSize = size
    #                 design.generateGeometry(reuse=False, headless=True)
    #                 # flag = True
    #                 for size in [0.35, 0.3, 0.25]:
    #                     try:
    #                         design.generateMesh(size, reuse=False, nprocessors=12)
    #                         design.homogenizationElastic.run(nprocessors=12, reuse=False, cases='all')
    #                         design.homogenizationElastic.process(check=True, reuse=False)
    #                         flag = True
    #                         break
    #                     except:
    #                         continue
    #                 if flag:
    #                     break
    #             except:
    #                 f.write(f"Design {design} failed\n")
    #                 f.flush()
    #                 continue

    #         # try:
    #         #     # check = design.checkMesh(0.15)
    #         #     # print(check)
    #         #     # design.homogenizationElastic.appliedStrain = 0.01
    #         #     # check = design.homogenizationElastic.check()
    #         #     # cases = [k for k, v in check.items() if v == False]
    #         #     design.homogenizationElastic.run(nprocessors=12, reuse=False, cases='all')
    #         #     design.homogenizationElastic.process(check=True, reuse=False)
    #         # except:
    #         #     f.write(f"Design {design} failed: {design.homogenizationElastic.check()}\n")
    #         #     f.flush()
    #         #     pass
    #         break
        # break
    # design.homogenizationElastic.plotE()
    # print(failed)
    # design.export()
    # design.generateGeometry(headless=True, reuse=False)
    # design.generateMesh(elementSize=0.5, reuse=False, nprocessors=8,
    #                     cleanup=True)
    # ps = design.homogenizationElastic.run(blocking=True, reuse=False,
    #                                       nprocessors=8)
    # check = design.homogenizationElastic.check()
    # design.homogenizationElastic.process(check=False, reuse=False)
    # print(design.homogenizationElastic)
    print(design)
