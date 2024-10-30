from unitcellengine.analysis.internal import homogenization, isotropicC, isotropicK
from unitcellengine.analysis.material import Ei, Ki, Gij, nuij, _iext, _ijext, _Eext, _d, _n
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import logging
from unitcellengine.utilities import timing
import re
import json
from _ctypes import PyObj_FromPtr
import pprint as pp
from numba import jit, prange
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# I primarily followed these references for the elastic homogenization
# calculations: 
# - Pinho-da-Cruz, J., Oliveira, J. A., & Teixeira-Dias, F. (2009).
#   Asymptotic homogenisation in linear elasticity. Part I: Mathematical
#   formulation and finite element modelling. Computational Materials
#   Science, 45(4), 1073–1080.
#   https://doi.org/10.1016/J.COMMATSCI.2009.02.025
# - Yi, S., Xu, L., Cheng, G., & Cai, Y. (2015). FEM formulation of
#   homogenization method for effective properties of periodic
#   heterogeneous beam and size effect of basic cell in thickness
#   direction. Computers & Structures, 156, 1–11.
#   https://doi.org/10.1016/J.COMPSTRUC.2015.04.010 
#
# Note that most homogenization papers are discussed in the context of
# elasticity theory and doesn't provide much insight into the actual
# implementation. For those that do discuss the implementation, most do
# so in a way that is not solveable in the context of commercial
# software. The second reference above is the only reference that I came
# across that does both rigorous asymptotic homogenization (rather that
# basic averaging based homogenization) in the context of commercial
# software. 


logger = logging.getLogger(__name__)

# Create the von Mises matrices
V = np.zeros((6, 6))
V[0, 0] = V[1, 1] = V[2, 2] = 1
V[3, 3] = V[4, 4] = V[5, 5] = np.sqrt(6)
V[0, 1] = V[1, 2] = V[2, 0] = -1
V2 = np.matmul(V.T, V)


@jit(nopython=True, cache=True)
def _worstStresses(A):
    """ Calculate worst case element stresses from amplification matrix """
    
    # Calculate M = A^T V^T V A, where A is the stress amplification
    # matrix, which corresponds to the square of the local von mises
    # stress. In a sense, it is the von Mises amplification matrix.
    P = A.shape[0]
    maxStresses = np.zeros(P)
    maxDirections = np.zeros((A.shape[1], P))
    # Ms = (A.transpose((0, 2, 1)) @ V2 @ A).astype(np.complex128)
    for i in range(P):
        # Calculate M
        # Note that we copy A in here to create contiguous arrays (in
        # memory) that are then faster to operate upon
        subA = A[i, :, :].copy()
        subAT = (subA.T).copy()
        M = (subAT @ V2 @ subA)
        
        # Solve the eigenvalue problem for each element
        # Here, the matrix is converted to a complex form to prevent 
        # numerical issues that can occur when the determinant of M is 
        # near zero and slightly negative. The converstion is only 
        # necessary when using numba, which can't handle the shift from 
        # real to complex.
        lams, vs = np.linalg.eig(M.astype(np.complex128))
        lams = np.real(lams)
        vs = np.real(vs)

        # Store the max values
        worstInd = np.argmax(lams)
        maxStresses[i] = np.sqrt(0.5*lams[worstInd])
        maxDirections[:, i] = vs[:, worstInd]

    # maxInd = np.argmax(maxStresses)

    return maxStresses, maxDirections

# Class used to prevent indentation of certain objects within a JSON
# structure 
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value

def numpy2str(data):
    ''' Use pprint to generate a nicely formatted string
    '''

    # Get rid of array(...) and keep only [[...]]
    f = pp.pformat(data, width=sys.maxsize)
    f = f[6:-1].splitlines() # get rid of array(...) and keep only [[...]]

    # Remove identation caused by printing "array(" 
    for i in range(1,len(f)):
        f[i] = f[i][6:]

    return '\n'.join(f)

# Define encoders/decoders for JSON data
class NumpyEncoder(json.JSONEncoder):

    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super().__init__(**kwargs)

    # def default(self, obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     return json.JSONEncoder.default(self, obj)
    

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, np.ndarray)
                else super().default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super().encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.tolist(), sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr


# class NumpyDecoder(json.JSONDecoder):
#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
#     def object_hook(self, dct):
#         for k, v in dct.items:
#             if isinstance(v, np.object):
#         if isinstance(obj, list):
#             return np.array(obj)
#         return obj


def pMean(values, p):
    """ p-mean metric, which is lower bound on the local maximum 
    
    Inputs
    ------
    values: list-like of length N
        Values to be aggregated with the p-mean metric
    p: int > 1
        P exponent value. As p -> infinity, p-mean -> maximum value of
        *values* from below.


    Theory
    ------
    p-mean = (1/N sum(f^p))^(1/p)
    """
    N = len(values)
    return ((values**p).sum()/N)**(1./p)

def pNorm(values, p):
    """ p-norm metric, which is an upper bound on the local maximum 
    
    Inputs
    ------
    values: list-like of length N
        Values to be aggregated with the p-norm metric
    p: int > 1
        P exponent value. As p -> infinity, p-norm -> maximum value of
        *values* from above.


    Theory
    ------
    p-norm = (sum(f^p))^(1/p)
    """
    N = len(values)
    return ((values**p).sum())**(1./p)

def upperKS(values, p):
    """ Upper KS metric, which is an upper bound on the local maximum 
    
    Inputs
    ------
    values: list-like of length N
        Values to be aggregated
    p: int > 1
        P exponent value. As p -> infinity, upper KS-> maximum value of
        *values* from above.


    Theory
    ------
    upper KS = ln(sum(exp(P f)))/P
    """
    N = len(values)
    return np.log(np.exp(values*p).sum())/p

def lowerKS(values, p):
    """ Lower KS metric, which is a lower bound on the local maximum 
    
    Inputs
    ------
    values: list-like of length N
        Values to be aggregated
    p: int > 1
        P exponent value. As p -> infinity, lower KS-> maximum value of
        *values* from below.


    Theory
    ------
    lower KS = upper KS - ln(N)/p
    """
    N = len(values)
    return upperKS(values, p) - np.log(N)/p

def readExodusNames(dataset, name):
    """ Read in name definitions for a given Exodus parameter """

    assert name[:5] == 'name_', f"Invalid Exodus name {name}. First " +\
                               "characters must be 'name_'"

    # Parse the global variable names, which are stored in a masked
    # numpy array with each element corresponding to a byte char of the
    # variable name
    bnames = dataset.variables[name]
    names = []
    for bname in bnames:
        # Pull out valid data
        subbname = bname[~bname.mask]
        names.append("".join([f"{c.decode('utf-8')}" for c in subbname]))
    
    return names

def readExodusValues(dataset, name):
    """ Read in variable definitions for a given exodus parameter """

    assert name[:5] == 'vals_', f"Invalid Exodus name {name}. First " +\
                               "characters must be 'vals_'"
    
    # Convert all data from big edian to little endian as big edian is
    # not supported by pandas (or, at least the current version used
    # locally). See
    # https://stackoverflow.com/questions/60161759/valueerror-big-endian-buffer-not-supported-on-little-endian-compiler
    # for reference.
    data = dataset.variables[name][:]
    if data.dtype.byteorder == '>':
        data = data.byteswap().newbyteorder()
    return data


def readExodusGlobal(filename):
    """ Read global variables from ExodusII file 
    
    Arguments
    --------
    filename: str or Path
        Filename of ExodusII file.
    
    Returns
    -------
    Pandas DataFrame containing global variables

    """
    filename = Path(filename)

    # Make sure the input mesh is an exodus file
    assert filename.suffix in ['.e', '.exo'], \
        "Input mesh must be an ExodusII file. Specified file is of " +\
        f"type {filename.suffix}"

    # Read in exodus file
    with Dataset(filename, 'r') as e:
        names = readExodusNames(e, 'name_glo_var')
        data = readExodusValues(e, 'vals_glo_var')

    # Compile data into a pandas dataframe and return
    df = pd.DataFrame(data, columns=names)
    return df

def readExodusElemental(filename):
    """ Read elemental variables from ExodusII file 
    
    Arguments
    --------
    filename: str or Path
        Filename of ExodusII file.
    
    Returns
    -------
    Dictionary containing elemental data

    """
    filename = Path(filename)

    # Make sure the input mesh is an exodus file
    assert filename.suffix in ['.e', '.exo'], \
        "Input mesh must be an ExodusII file. Specified file is of " +\
        f"type {filename.suffix}"

    # Read in exodus file
    e = Dataset(filename, 'r')

    # Read in element variable names
    names = readExodusNames(e, 'name_elem_var')

    # Find relevant element keys (noting that, if there are multiple
    # integration point for an element, that data will be exported as a
    # unique variable for each integration point)
    keys = [k for k in e.variables.keys() if 'vals_elem' in k]
    
    # Read in each set of data and store it in a pandas dataframe
    data = {}
    for k, name in zip(keys, names):
        d = readExodusValues(e, k)
        data[name] = d

    e.close()

    return data

def readExodusNodal(filename):
    """ Read nodal variables from ExodusII file 
    
    Arguments
    --------
    filename: str or Path
        Filename of ExodusII file.
    
    Returns
    -------
    Dictionary containing nodal data

    """
    filename = Path(filename)

    # Make sure the input mesh is an exodus file
    assert filename.suffix in ['.e', '.exo'], \
        "Input mesh must be an ExodusII file. Specified file is of " +\
        f"type {filename.suffix}"

    # Read in exodus file
    e = Dataset(filename, 'r')

    # Read in nodal variable names
    names = readExodusNames(e, 'name_nod_var')

    # Find relevant element keys (noting that, if there are multiple
    # integration point for an element, that data will be exported as a
    # unique variable for each integration point)
    keys = [k for k in e.variables.keys() if 'vals_nod' in k]
    
    # Read in each set of data and store it in a pandas dataframe
    data = {}
    for k, name in zip(keys, names):
        d = readExodusValues(e, k)
        data[name] = d

    e.close()

    return data

# .


class Homogenization(object):
    """ Abstract class definition used to manage property homogenization """
    
    KIND = None
    N = None

    def __init__(self, mesh, path=None):
        """ Constructor for homogenization class 
        
        Arguments
        ---------
        mesh: str or Path
            Unitcell mesh to homogenize. This mesh needs to be created
            by the meshing submodule so that the necessary nodesets are
            generated.
        
        Keywords
        --------
        path: str or Path or None (Default=None)
            Defines the base path for all file generation. If none,
            files are located in the same folder as the input mesh file.
         

        """
        self.mesh = Path(mesh)
        self.result = None
        self.CH = None

        # Parse the path input
        if path:
            self.path = Path(path)
            assert self.path.is_dir(), f"Specified path {path} is not a " +\
                                       "directory."
        else:
            self.path = self.mesh.parent

        # Check to see if results already exist. If so, store in results
        # property
        if self.processed:
            self.loadResults()
            # self.result = np.genfromtxt(self.homogenizationFile)
            logger.info("Processed data already exists and has been "
                        f"loaded from file {self.homogenizationFile}")
    
    def __repr__(self):
        output = f"{self.__class__.__name__}({self.mesh})"
        # If results are present, print array
        try:
            for row in self.CH:
                output += "\n\t["
                output += ", ".join([f"{x:.3e}" for x in row])
                output += "]"
            output += "\n"
        except:
            output += " (Not processed)"

        return output
    
    def meshQuality(self):
        """ Return each element's quality metric """
        raise NotImplementedError("Need to implement 'meshQuality' method")

    @property
    def processed(self):
        """ Has the homogenization been processed yet? """
        if self.homogenizationFile.exists() or self.result:
            return True
        else:
            return False 
    
    @property
    def homogenizationFile(self):
        """ Homogenization filename for post processed data"""
        filename = self.path / Path(self.mesh.stem +\
                                    f"_{self.KIND}_homogenization.json")
        return filename

    def run(self, reuse=True, **kwargs):
        """ Run required simulations for homogenization """
        raise NotImplementedError("Need to implement 'run' method")
    
    def clear(self):
        """ Clear all outputs files that may exist """
        files = [self.homogenizationFile, self.result]
        for file in files:
            if file:
                file = Path(file)
                file.unlink(missing_ok=True)

    def preprocess(self, **kwargs):
        """ Preprocess the results to a common form
        
        Notes
        -----
        This method needs to minimally calculate the homogenized matrix 
        and store it in self.CH.
         """
        raise NotImplementedError("Need to implement 'preprocess' method")

    def process(self, rtol=1e-3, check=True, reuse=True, **kwargs):
        """ Calculate homogenization based on simulation results """

        # Reset the result if the result isn't to be reused
        if not reuse:
            self.result = None

        # If reuse is specified and the data already exists, break out
        # early 
        if reuse and self.processed:
            return
        
        # Preprocess data
        self.preprocess(**kwargs)

        # Pull out the min and max values for reference
        CH = self.CH
        CHmax = CH.max()
        CHmin = CH.min()

        # Define an absolute tolerance for comparisons based on the largest
        # value of the stiffness matrix.
        atol = rtol*(CHmax-CHmin)

        # Make sure all diagonals are possitive
        if check and any(np.diag(CH) < 0):
            raise RuntimeError("The computed homogenized matrix "
                               "has negative elements along the diagonal "
                               "which is aphysical. Something likely "
                               "went wrong with the homogenization "
                               "simulations. Check log files for poor "
                               "convergence or output files for large "
                               "mesh distortions.")

        # Check for rough symmetry
        if check and not np.allclose(CH, CH.T, atol=atol):
            raise RuntimeError("The computed homogenized matrix "
                               "is not symmetric, even within numerical "
                               "noise. Something likely went wrong "
                               "with the homogenization simulations. "
                               "Check log files for poor convergence "
                               "or output files for large mesh distortions.")
        
        # Force symmetry
        CH += CH.T
        CH /= 2

        # Convert essentially zero quantities to zero
        # CH[np.isclose(CH, 0., rtol=rtol, atol=atol)] = 0.

        # Store the homogenization result
        self.CH = CH

    def loadResults(self):
        """ Load pre-existing results """
        raise NotImplementedError("Need to implement 'loadResults' method ")


class ElasticHomogenization(Homogenization):
    """ Linear elastic homogenization 
    
    Note
    ----
    The calculated constitutive matrix is in Voigt notation using
    tensorial shear strain rather than engineering shear strain.
    """

    KIND = 'elastic'
    N = 6

    def __init__(self, mesh, E, nu, **kwargs):
        """ Elastic homogenization """

        # Check the constitutive input parameters
        assert E > 0, f"Elastic modulus must be greater than zero, not {E}"
        assert -1 < nu < 0.5, "Poisson's ratio must be greater than zero, " +\
                               f"must be between -1 and 0.5, not {nu}."
        self.E = E
        self.nu = nu
        self.maxStresses = None
        self.volumeIntStress = None
        self.volumeUnitcell = None
        self.volumeMaterial = None
        self.strains = None

        # Run the super class constructor
        super().__init__(mesh, **kwargs)

    
    
    def check(self):
        """ Check the simulation results for errors 
        
        Returns
        -------
        Dictionary with the run indeces as the keys and the run success
        represented as a boolean value.
        """

        raise NotImplementedError()
    
    def preprocess(self):
        """ Preprocess the results data
        
        This method should create the following properties:
            - CH: 6x6 homogenized stiffness matrix
            - displacement: nnodes x 3 x 6 array that defines the nodal
            displacement results for each homogenization load case 
            - strain: nelx6x6 array that defines the elemental
            average strain for each homogenization load case
            (column-wise)
        """
        
        raise NotImplementedError("Preprocess method not implemented yet")
    
    def loadResults(self):
        """ Load results file """

        # Run stock preprocessing
        self.preprocess()

        # Load the results json file
        with open(self.homogenizationFile, 'r') as f:
            result = json.load(f)
        
        # Load in the material behavior
        E = result['E']
        nu = result['nu']
        if not np.isclose(E, self.E):
            logger.warning("The loaded results file does not correspond "
                           "the to current elastic modulus value E of "
                           f"{self.E}. Ignoring results.")
            return

        if not np.isclose(nu, self.nu):
            logger.warning("The loaded results file does not correspond "
                           "the to current elastic modulus value of ν "
                           f"{self.nu}. Ignoring results.")
            return

        # Convert relevant lists to numpy arrays
        result['CH'] = np.array(result['CH'])
        result['SH'] = np.array(result['SH'])
        self.result = result

        # Pull out the homogenized elasticity matrix
        self.CH = result['CH'] 

        self.E = E
        self.nu = nu

    @property
    def C(self):
        """ Solid material isotropic elastic stiffness matrix (tensorial)"""

        # Calculate the relevant isotropic material proerpties
        C = isotropicC(self.E, self.nu)

        # Convert from engineering to tensorial strains
        C[:, 3:] *= 2

        return C

    @property
    def SH(self):
        """ Homogenized compliance matrix """
        return np.linalg.inv(self.CH)

    @property
    def SHtensor(self):
        """ Homogenized compliance tensor 
        
        Note that you need to be careful in doing this processes as it
        depends on the Voigt notation form being used.
        """

        # Convert C to standard Voigt notation prior to calculating the
        # compliance matrix
        C = self.CH.copy()
        C[:, 3:] = C[:, 3:]/2
        S = np.linalg.inv(C)

        # Create the compliance tensor
        St = np.ones((3, 3, 3, 3))
        mapping = [(0, 1), (1, 2), (2, 0)]
        for i in range(3):
            for j in range(3):
                St[i, i, j, j] = St[j, j, i, i] = S[i, j]
            
            for j, (m, n) in zip(range(3, 6), mapping):
                St[i, i, m, n] = St[m, n, i, i] = S[i, j]/2
                St[i, i, n, m] = St[n, m, i, i] = S[i, j]/2

        for i, (m, n) in zip(range(3, 6), mapping):
            for j, (p, q) in zip(range(3, 6), mapping):
                St[q, p, m, n] = St[p, q, n, m] = St[n, m, q, p] = S[i, j]/4
                St[m, n, p, q] = St[n, m, p, q] = St[m, n, q, p] = St[p, q, m, n] = S[i, j]/4
        
        return St
                


    @timing(logger)
    def process(self, save=True, reuse=True, rtol=1e-3, check=True, **kwargs):
        logger.info("Processing elastic homogenization results.")


        # Run the parent class process method
        super().process(reuse=reuse, rtol=rtol, check=check, **kwargs)
        

        # Calculate local strain mappinging matrices, which is a
        # (nel x 6 x 6) matrix (noting that it is in this form due to
        # how matmul works in numpy - i.e, matmul on ND arrays
        # corresponds to the matrix multiplication of the last 2 dims or
        # each array).
        # From Asymptotic homogenisation in linear elasticity. 
        # Part I: Mathematical formulation and finite element modelling. 
        # Computational Materials Science, 45(4), 1073–1080. 
        # https://doi.org/10.1016/J.COMMATSCI.2009.02.025
        # eps(x, y) = A*eps_0(x)
        self.strainAmplification = np.eye(self.N)[np.newaxis] - self.strains


        # Create a dictionary with all of the output data
        self.result = dict(CH=self.CH, SH=self.SH, 
                           anisotropyIndex=self.anisotropyIndex,
                           E=self.E, nu=self.nu)
        
        # Calculate bounding engineering constants
        Eext = self.Eext()
        Gext = self.Gext()
        Kext = self.Kext()
        nuext = self.nuext()
        self.result['engineeringConstants'] = \
                dict(Emax=Eext['max'], Emin=Eext['min'],
                     Kmax=Kext['max'], Kmin=Kext['min'],
                     Gmax=Gext['max'], Gmin=Gext['min'],
                     numax=nuext['max'], numin=nuext['min'],)

        # Determine the mesh quality and apply a mask to pull only good
        # quality elements
        quality = self.meshQuality()
        goodQuality = quality > 0.2

        # Calculate the stress amplification values and worst case
        # stresses
        unitApplications = dict(vm11=None, vm22=None, vm33=None, 
                                vm12=None, vm23=None, vm13=None,
                                vmWorst=None)
        for i, name in enumerate(['vm11', 'vm22', 'vm33',
                                  'vm12', 'vm23', 'vm13']):
            # Define unit stress vector
            astress = np.zeros(6)
            astress[i] = 1.

            # Calculate von Mises stresses
            vms = self.localVMStress(astress)

            # Aggregate the stresses to remove spurious peaks that may
            # arise from stress concentrators or bad elements
            unitApplications[name] = vms[goodQuality, 0].max()

        # Calculate the worst case stress amplication and corresponding
        # loading direction
        maxvms, maxvmDirs = self.processWorstStress()
        maxInd = np.argmax(maxvms[goodQuality])
        unitApplications['vmWorst'] = (maxvms[goodQuality][maxInd],
                                       maxvmDirs[:, goodQuality][:, maxInd])

        self.result['amplification'] = unitApplications

        # Save results if requested
        if save:
            # Save the data to a json file
            with open(self.homogenizationFile, 'w') as f:
                f.write(json.dumps(self.result, cls=NumpyEncoder, indent=2))

        return self.result
    
    def Ei(self, d):
        """ Calculate the elastic modulus in a specific direction 
        
        Arguments
        ---------
        d: length 3 array-like
            Primary direction

        Theory
        ------
        E(d) = 1/(d ⨂ d : S : d ⨂ d)

        Note
        ----
        The C matrix calculated in this homogenization process is
        different than the C matrix defined in the below reference.
        
        References
        ----------
        Nordmann, J., Aßmus, M., & Altenbach, H. (2018). Visualising
        elastic anisotropy: theoretical background and computational
        implementation. Continuum Mechanics and Thermodynamics, 30(4),
        689–708. https://doi.org/10.1007/s00161-018-0635-9 
        
        """

        return Ei(self.CH, d)
    
    def Eext(self):
        """ Calculate the extreme values for the elastic modulus """

        # Pull out the compliance tensor
        St = self.SHtensor

        # Execute jit compiled function for faster run time
        outmin, outmax = _Eext(St)

        # Export in a more readable format
        return dict(max=dict(value=outmax[0], d=outmax[1]), 
                min=dict(value=outmin[0], d=outmin[1]))

        # return _iext(self.Ei)
        
    def plotE(self):
        """ Plot 3D variation of elastic modulus """

        # Create a grid to plot the elastic modulus on
        N = 40
        M = int(N/2)
        PHI, THETA = np.meshgrid(np.linspace(0, np.pi, N),
                                 np.linspace(0, 2*np.pi, M))
        
        # Calculate the elastic modulus at each grid point and map it
        # into cartesian space
        E = np.zeros((M, N)) 
        X = np.zeros((M, N))
        Y = np.zeros((M, N))
        Z = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                dij = _d(PHI[i, j], THETA[i, j])
                E[i, j] = self.Ei(dij)
                X[i, j], Y[i, j], Z[i, j] = E[i, j]*dij
        
        # Plot the geometry
        surface = go.Surface(x=X, y=Y, z=Z, surfacecolor=E)
        fig = go.Figure(data=surface)
        fig.update_traces(contours_x=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_x=True),
                        contours_y=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_y=True),
                        contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True))
        fig.show()

    def Ki(self, d):
        """ Calculate the bulk modulus in a specific direction 
        
        Arguments
        ---------
        d: length 3 array-like
            Primary direction

        Theory
        ------
        K(d) = 1/(I : S : d ⨂ d)

        Note
        ----
        The C matrix calculated in this homogenization process is
        different than the C matrix defined in the below reference.
        
        References
        ----------
        Nordmann, J., Aßmus, M., & Altenbach, H. (2018). Visualising
        elastic anisotropy: theoretical background and computational
        implementation. Continuum Mechanics and Thermodynamics, 30(4),
        689–708. https://doi.org/10.1007/s00161-018-0635-9 
        
        """

        return Ki(self.CH, d)
    
    def plotK(self):
        """ Plot 3D variation of elastic modulus """

        # Create a grid to plot the elastic modulus on
        N = 40
        M = int(N/2)
        PHI, THETA = np.meshgrid(np.linspace(0, np.pi, N),
                                 np.linspace(0, 2*np.pi, M))
        
        # Calculate the elastic modulus at each grid point and map it
        # into cartesian space
        K = np.zeros((M, N)) 
        X = np.zeros((M, N))
        Y = np.zeros((M, N))
        Z = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                dij = _d(PHI[i, j], THETA[i, j])
                K[i, j] = self.Ki(dij)
                X[i, j], Y[i, j], Z[i, j] = K[i, j]*dij
        
        # Plot the geometry
        surface = go.Surface(x=X, y=Y, z=Z, surfacecolor=K)
        fig = go.Figure(data=surface)
        fig.update_traces(contours_x=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_x=True),
                        contours_y=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_y=True),
                        contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True))
        
        return fig

    def Kext(self):
        """ Calculate the max and min bulk modulus values """

        # Pull out the compliance tensor
        St = self.SHtensor

        # Solve the 2nd order tensor eigenvalue problem
        eig, eigv = np.linalg.eig(np.einsum('ij,ijkl->kl', np.eye(3)*3, St))
        imax = np.argmax(eig)
        imin = np.argmin(eig)

        # Take the inverse to calculate the bulk modulus values
        Kmax = 1/eig[imin]
        dmax = eigv[imin, :]
        Kmin = 1/eig[imax]
        dmin = eigv[imax, :]
        
        return dict(max=dict(value=Kmax, d=dmax), 
                    min=dict(value=Kmin, d=dmin))

    def Gij(self, *args):
        """ Calculate the bulk modulus in a specific orientation 
        
        Arguments
        ---------
        d: length 3 array-like
            Primary direction
        n: length 3 array-like
            Normal direction

        OR

        phi: 0 <= float <= pi
            Loading direction polar angle in radians (x-y plane)
        theta: 0 <= float <= 2pi
            Loading direction azimuth angle in radians
        psi: 0 <= float <= 2pi
            Normal vector rotation angle in radians

        Theory
        ------
        G(d, n) = 1/((d ⨂ n + n ⨂ d ) : S : (d ⨂ n + n ⨂ d ))

        d = [sin(φ)cos(θ),sin(φ)sin(θ),cos(φ)]
        n = [sin(θ)cos(ψ)-cos(φ)cos(θ)cos(ψ),
             -cos(θ)sin(ψ)-cos(φ)sin(θ)cos(ψ)
             sin(φ)cos(ψ)]

        Note
        ----
        The C matrix calculated in this homogenization process is
        different than the C matrix defined in the below reference.
        
        References
        ----------
        Nordmann, J., Aßmus, M., & Altenbach, H. (2018). Visualising
        elastic anisotropy: theoretical background and computational
        implementation. Continuum Mechanics and Thermodynamics, 30(4),
        689–708. https://doi.org/10.1007/s00161-018-0635-9 
        
        """

        return Gij(self.CH, *args)
    
    def plotG(self):
        """ Plot 3D variation of elastic modulus """

        # Create a grid to plot the elastic modulus on
        N = 40
        M = N*2
        PHI, THETA = np.meshgrid(np.linspace(0, np.pi, N),
                                 np.linspace(0, 2*np.pi, M))
        # psis = np.linspace(0, np.pi/2, P)
        psis = np.arange(0, 181, 10)
        P = len(psis)

        # Calculate the shear modulus at each grid point and map it
        # into cartesian space, adding a slider in to explore different
        # orientations 
        Gs = np.zeros((M, N, P)) 
        X = np.zeros((M, N, P))
        Y = np.zeros((M, N, P))
        Z = np.zeros((M, N, P))
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Full", "Min", 'Max'),
                            specs=[[{'type': 'surface'}]*3])
        steps = []
        for k, psi in enumerate(psis):
            for i in range(M):
                for j in range(N):
                    dij = _d(PHI[i, j], THETA[i, j])
                    Gs[i, j, k] = self.Gij(dij, _n(PHI[i, j], 
                                               THETA[i, j], 
                                               psi*np.pi/180))
                    X[i, j, k], Y[i, j, k], Z[i, j, k] = Gs[i, j, k]*dij
            fig.add_trace(go.Surface(visible=False,
                                     x=X[:, :, k], y=Y[:, :, k], z=Z[:, :, k], 
                                     surfacecolor=Gs[:, :, k],
                                     customdata=Gs[:, :, k],
                                     hovertemplate="%{customdata[0]:.4f}",
                                     coloraxis="coloraxis"),
                                     row=1, col=1)
            
            step = dict(
                        method="update",
                        args=[{"visible": [False]*P + [True]*2},
                            {"title": f"Angle: {psi}°"}],  # layout attribute
                    )
            step["args"][0]["visible"][k] = True  # Toggle i'th trace to "visible"
            steps.append(step)
        # Make initial trace visible
        fig.data[0].visible = True

        # Definet the normal vector slider
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Angle: "},
            pad={"t": 50},
            steps=steps
        )]    
        

        # Plot the min and max G fields
        minargs = np.expand_dims(np.argmin(Gs, axis=2), axis=2)
        maxargs = np.expand_dims(np.argmax(Gs, axis=2), axis=2)
        for i, inds in zip(range(2, 4), 
                                 [minargs, maxargs]):
            Xext = np.take_along_axis(X, inds, axis=2)[:, :, 0] 
            Yext = np.take_along_axis(Y, inds, axis=2)[:, :, 0] 
            Zext = np.take_along_axis(Z, inds, axis=2)[:, :, 0] 
            Gext = np.take_along_axis(Gs, inds, axis=2)[:, :, 0] 
            
            fig.add_trace(go.Surface(x=Xext, y=Yext, z=Zext, 
                                     surfacecolor=Gext,
                                     customdata=Gext,
                                     hovertemplate="%{customdata[0]:.4f}",
                                     coloraxis="coloraxis"),
                                     row=1, col=i)


        # Plot the geometry
        fig.update_traces(contours_x=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_x=True),
                        contours_y=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_y=True),
                        contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True))
        
        fig.update_layout(
            sliders=sliders
        )
        
        return fig

    def Gext(self):
        """ Calculate the extreme values for the shear modulus """

        return _ijext(self.Gij)

        # out = {}
        # for sign, name in zip([1, -1], ['min', 'max']):
        #     fun = lambda x: sign*self.Gij(*x)
        #     sol = optimize.differential_evolution(fun, [(0, np.pi),
        #                                                 (0, 2*np.pi),
        #                                                 (0, 2*np.pi)])
        #     assert sol['success'], f"Searching for the {name} shear " +\
        #                             "modulus failed."
        #     phi, theta, psi = sol['x']
        #     d = _d(phi, theta)
        #     n = _n(phi, theta, psi)
        #     out[name] = dict(value=sign*sol['fun'], d=d, n=n)
        
        # return out

        # # Pull out the compliance tensor
        # St = self.SHtensor
        
        # #  Run jit compliled function
        # outmin, outmax = _Gext(St)

        # # Export in a more readable format
        # return dict(max=dict(value=outmax[0], d=outmax[1], n=outmax[2]),
        #             min=dict(value=outmin[0], d=outmin[1], n=outmin[2]))

    def nuij(self, *args):
        """ Calculate the poissons ratio in a specific orientation 
        
        Arguments
        ---------
        d: length 3 array-like
            Primary direction
        n: length 3 array-like
            Normal direction

        OR

        phi: 0 <= float <= pi
            Loading direction polar angle in radians (x-y plane)
        theta: 0 <= float <= 2pi
            Loading direction azimuth angle in radians
        psi: 0 <= float <= 2pi
            Normal vector rotation angle in radians

        Theory
        ------
        ν(d, n) = -((d ⨂ d) : S : (n ⨂ n))/((d ⨂ d) : S : (d ⨂ d))

        d = [sin(φ)cos(θ),sin(φ)sin(θ),cos(φ)]
        n = [sin(θ)cos(ψ)-cos(φ)cos(θ)cos(ψ),
             -cos(θ)sin(ψ)-cos(φ)sin(θ)cos(ψ)
             sin(φ)cos(ψ)]

        Note
        ----
        The C matrix calculated in this homogenization process is
        different than the C matrix defined in the below reference.
        
        References
        ----------
        Nordmann, J., Aßmus, M., & Altenbach, H. (2018). Visualising
        elastic anisotropy: theoretical background and computational
        implementation. Continuum Mechanics and Thermodynamics, 30(4),
        689–708. https://doi.org/10.1007/s00161-018-0635-9 
        
        """

        return nuij(self.CH, *args)
    def nuext(self):
        """ Calculate the extreme values for Poisson's ratio """

        return _ijext(self.nuij)

    @property
    def stressAmplification(self):
        """ Local stress amplification matrix (6 x 6 x nel)

        sig(x, y) = A*sig_mean(x)

        Returns
        -------
        nel x 6 x 6 numpy array corresponding to the mean to local
        amplification matrix for each unit cell element
        """
        assert self.processed

        # Calculate the amplification matrix A (N_el x 6 x 6)
        # A = Ce M SH, where M =  (I - eps*) is the strain amplification
        # matrix
        return self.C @ (self.strainAmplification @ self.SH)

    def localStress(self, averageStress):
        """ Calculate the local stress based on the average stress 
        
        Arguments
        ---------
        averageStress: len(6) or 6xN array-like
            Average stress at the macroscopic scale for N load cases.
        
        Returns
        -------
        nel x 6 x N array containing the elemental stresses within the unit
        cell
        """
        
        # Make sure the input is an array
        averageStress = np.array(averageStress)

        assert self.processed
        assert averageStress.size % self.N == 0

        # Make sure the shape is correct
        if averageStress.shape[0] != self.N:
            averageStress = averageStress.T
        averageStress = averageStress.reshape((self.N, -1))

        # sig(x, y) = A*sig_0(x)
        return (self.stressAmplification @ averageStress)
    
    def localVMStress(self, averageStress):
        """ Calculate the local von mises stress based on the average stress 
        
        Arguments
        ---------
        averageStress: len(6) or 6xN array-like
            Average stress at the macroscopic scale for N different load cases.
        
        Returns
        -------
        nel x N array containing the elemental von mises stresses within the unit
        cell
        """
        
        # Calculate local stresses
        stresses = self.localStress(averageStress)

        # vm = sqrt(0.5 σ^T V2 σ)
        vm2 = 0.5*(stresses * (V2[None, :, :] @ stresses)).sum(axis=1)
        return np.sqrt(vm2)
        # return np.array(list(map(lambda x: np.sqrt(0.5*(x[0] @ x[1].T)), 
        #                             zip(stresses.T, (V2 @ stresses).T))))
    
    def localStrain(self, averageStrain):
        """ Calculate the local strain based on the average strain 
        
        Arguments
        ---------
        averageStrain: len(6) or 6x1 array-like
            Average strain at the macroscopic scale
        
        Returns
        -------
        6 x nel array containing the elemental strains within the unit
        cell
        """
        assert self.processed
        assert len(averageStrain) == self.N

        # Make sure the input is the correct shape
        if averageStrain.shape == [self.N]:
            averageStrain = averageStrain.reshape(self.N, 1)
        elif averageStrain.shape == [1, self.N]:
            averageStrain = averageStrain.T

        # eps(x, y) = A*eps_0(x)
        return (self.strainAmplification @ averageStrain).T
    
    @timing(logger)
    def processWorstStress(self):
        """ Determine the worst possible load orientation 
        
        
        Theory
        ------
        The von Mises calculation can be represented by the following
        calculation: sig_vm = sqrt(0.5 sig^T V^t V sig), where sig is the
        6x1 stress state, 
        V = [ 1 -1  0  0  0  0]
            [ 0  1 -1  0  0  0]
            [-1  0  1  0  0  0]
            [ 0  0  0 s6  0  0]
            [ 0  0  0  0 s6  0]
            [ 0  0  0  0  0 s6]
        and s6 = sqrt(6)
        """

        assert self.processed
        

        # Calculate M = A^T V^T V A, where A is the stress amplification
        # matrix, which corresponds to the square of the local von mises
        # stress. In a sense, it is the von Mises amplification matrix.
        A = self.stressAmplification
        # P = A.shape[0]
        # maxStresses = np.zeros(P)
        # maxDirections = np.zeros((self.N, P))
        # for i, subA in enumerate(A):
        #     # Calculate M
        #     M = subA.T @ V2 @ subA
            
        #     # Solve the eigenvalue problem for each element
        #     lams, vs = np.linalg.eig(M)

        #     # Store the max values
        #     worstInd = np.argmax(lams)
        #     maxStresses[i] = np.sqrt(0.5*lams[worstInd])
        #     maxDirections[:, i] = vs[:, worstInd]

        # # maxInd = np.argmax(maxStresses)

        # return maxStresses, maxDirections

        return _worstStresses(A)
     
    @property
    def anisotropyIndex(self):
        """ Anisotropy index of the unit cell geometry 
        
        Theory
        ------
        This metric comes from Ranganathan, S. I., & Ostoja-Starzewski,
        M. (2008). Universal elastic anisotropy index. Physical Review
        Letters, 101(5), 3–6.
        https://doi.org/10.1103/PhysRevLett.101.055504.

        AU = 0 indicates isotropy
        As AU increases, the anisotropy increases
        """
        # @todo Validate anisotropyIndex
        # See https://wiki.materialsproject.org/Elasticity_calculations
        # for reference

        try:
            # Check to see if the result exists
            C = self.CH.copy()
        except AttributeError:
            raise AttributeError("No existing homogenization results. "
                               "Anisotropy index could not be "
                               "calculated.")

        # Note that the below calculations correspond to the elasticity
        # matrix defined with respect to engineering shear strain, while
        # calculated elasticity matrix is with respect to tensorial
        # shear strain. So, the matrix needs to be converted.
        # Additionally, the Voigt ordering is 11, 22, 33, 23, 13, 12
        # while the calculated Voigt ordering is 11, 22, 33, 12, 23, 13;
        # so, the rows and columns need to be swapped
        C[:, 3:] *= 0.5
        C[:, 3:] = np.roll(C[:, 3:], 1, axis=1)
        C[3:, :] = np.roll(C[3:, :], 1, axis=0)

        # Calculate the compliance matrix
        S = np.linalg.inv(C)

        # Pull out relevant stiffness values
        inds = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], 
                [5, 5], [0, 1], [1, 2], [2, 0]]
        Is = [i[0] for i in inds]
        Js = [j[1] for j in inds]
        C11, C22, C33, C44, C55, C66, C12, C23, C31 = C[Is, Js]
        S11, S22, S33, S44, S55, S66, S12, S23, S31 = S[Is, Js]

        # Calculate the isotropy index
        KV = (C11+C22+C33)+2*(C12+C23+C31)
        KV /= 9
        KR = 1/((S11+S22+S33)+2*(S12+S23+S31))
        GV = (C11+C22+C33) - (C12+C23+C31) + 3*(C44+C55+C66)
        GV /= 15
        GR = 15/(4*(S11+S22+S33) - 4*(S12+S23+S31) + 3*(S44+S55+S66))
        AU = 5*GV/GR+KV/KR-6

        return AU
    
    # def saveResults(self):
    #     """ """
    #     assert self.processed

class InternalHomogenization(object):
    """ Parent homogenization class for internal calculations """

    KIND: str
    mesh: str | Path 

    # def __init__(self):

    #     # Make sure the input mesh is an exodus file
    #     assert self.mesh.suffix == '.npz', \
    #         ("Input mesh must be a numpy npz file. Specified file is of "
    #         f"type {self.mesh.suffix}")

    #     # Read the mesh file and pull out the relevant dimensions
    #     with np.load(self.mesh) as data:
    #         ns = data['nodes']
    #     self.length, self.width, self.height = \
    #             ns[:, 1:].max(axis=0) - ns[:, 1:].min(axis=0)

    def meshQuality(self):

        # The mesh quality for the voxel mesh is ideal. However, 
        # very low density results may lead to bad stress/strain 
        # calculations, so define a mesh quality based on the element
        # density.
        with np.load(self.mesh) as data:
            rhos = data['erhos']
            einds = data['einds']

        # The quality here is defined as the element density offset by
        # 0.1.
        # @TODO Re-example this mesh quality definition for voxel mesh
        return rhos[einds] + 0.1

    @property
    def _resultfile(self):
        """ Get result filename """
        return self.path / Path(self.mesh.stem + "_" + self.KIND + "_results.npz")

    def check(self):
        """ Check the status of previous simulation runs """
        logger.debug("Check status of previous simulations.")
        if self._resultfile.exists():
            logger.debug("Results file found.")
            return True
        else:
            logger.debug("Results file not found.")
            return False

    def cleanup(self):
        """ Clean up results files """

        logger.debug("Cleaning up old results files.")
        basename = self._resultfile
        for suffix in ['.npz', '.vtu']:
            filename = basename.with_suffix(suffix)
            if filename.exists():
                logger.debug(f"Removing old results file {filename}")
                filename.unlink()

    @timing(logger)
    def run(self, blocking=True, reuse=True, nprocessors=True, cases='all',
            rtol=1e-6, atol=1e-8, solver='auto'):
        logger.info(f"Running {self.KIND} homogenization simulations")
        # If reuse is specified, check to see if outputs already exist
        checks = self.check()
        if reuse:
            logger.info("Reuse of homogenization results requested.")
            if checks:
                logger.info("All simulation result files exit and completed "
                            " successfully. No need to run any additional "
                            "simulations. Exiting early.")
                # p = subprocess.Popen("echo 'Process complete'",
                # shell=True)
                return True
            else:
                # Rerun all simulations
                logger.info("No previous simulations or previous"
                            "simulations failed. Running.")
        else:
            # If not, clear out old files
            logger.info("Removing old files prior to new simulation runs.")
            self.cleanup()
        
            
        # Solvenecessary simulations
        try:
            C = np.copy(self.C)
            if self.KIND == 'elastic':
                # Convert from tensorial stiffness matrix to engineering
                C[:, 3:] /= 2
            homogenization(self.mesh, C, 
                            self._resultfile, rtol=rtol, atol=atol,
                            parallel=nprocessors, solver=solver)
        except RuntimeError as ex:
            logger.error(f"{self.KIND} homogenization failed: {ex}")
            return False

        return True

class InternalElasticHomogenization(InternalHomogenization, 
                                    ElasticHomogenization):
    """ Homogenization based on internal finite element analysis """
    
    def preprocess(self):
        # Aggregate the homogenization data. There are a total of 6 runs
        # that are necessary for the elastic homogenization. Read in
        # each file and pull out the precomputed elasticity
        # coefficients. 

        # Check simulation results
        check = self.check()
        if not check:
            text = ("The homogenization runs failed. Homogenization "
                    "results cannot be processed.")
            logger.error(text)
            raise RuntimeError(text)

        # Read in the results
        with np.load(self._resultfile.with_suffix('.npz')) as data:
            CH = data['CH']
            disps = np.array([data[f'xi{i+1}'] for i in range(self.N)])
            epss = np.array([data[f'Bxi{i+1}'] for i in range(self.N)])

        # Convert CH from engineer to tensorial shear strains
        CH[:, 3:] *= 2

        # Store result
        self.CH = CH

        # Store strain data as an nel x 6 x 6 matrix
        self.strains = np.transpose(epss, (2, 0, 1))

        # Store displacement data as nnodes x 3 x 6 matrix
        self.displacements = np.transpose(disps, (2, 0, 1))

class ConductanceHomogenization(Homogenization):
    """ Linear conductance homogenization 
    
    """

    KIND = 'conductance'
    N = 3

    def __init__(self, mesh, k, **kwargs):
        """ Conductance homogenization """

        # Check the constitutive input parameters
        assert k > 0, f"Thermal conductance must be greater than zero, not {k}"
        self.k = k

        # Run the super class constructor
        super().__init__(mesh, **kwargs)
    
    def loadResults(self):
        """ Load results file """

        # Run stock preprocessing
        self.preprocess()
        
        # Load the results json file
        with open(self.homogenizationFile, 'r') as f:
            result = json.load(f)
        
        # Convert relevant lists to numpy arrays
        result['CH'] = np.array(result['CH'])
        self.result = result

        # Pull out the homogenized elasticity matrix
        self.CH = result['CH']
        
        # Load in the material behavior
        k = result['k']
        if not np.isclose(k, self.k):
            logger.warning("The loaded results file does not correspond "
                           "the to current conductance value k of "
                           f"{self.k}. Overwriting this property with the "
                           f"value corresponding to the results file: {k}.")
                           
                           
        self.k = k

    @property
    def C(self):
        """ Solid material isotropic conductance matrix """
        return isotropicK(self.k)      

    def kext(self):
        """ Calculate the extreme values for the conductance """

        # Calculate the eigenvalues and vectors of the conductance
        # matrix
        lams, ds = np.linalg.eig(self.CH)

        maxind = np.argmax(lams)
        minind = np.argmin(lams)

        # Export in a more readable format
        return dict(max=dict(value=lams[maxind], d=ds[:, maxind]), 
                min=dict(value=lams[minind], d=ds[:, minind]))

    @timing(logger)
    def process(self, save=True, reuse=True, rtol=1e-3, check=True, **kwargs):
        logger.info("Processing conduction homogenization results.")

        # Run superclass processing on the homogenization matrix
        super().process(reuse=reuse, rtol=rtol, check=check, **kwargs)

        # Create a dictionary with all of the output data
        self.result = dict(CH=self.CH, k=self.k)
        
        # # Calculate bounding engineering constants
        # Kext = self.Kext()
        # self.result['engineeringConstants'] = \
        #         dict(Kmax=Kext['max']['value'], Kmin=Kext['min']['value'])
  

        # Save results if requested
        if save:
            # Save the data to a json file
            with open(self.homogenizationFile, 'w') as f:
                f.write(json.dumps(self.result, cls=NumpyEncoder, indent=2))

        return self.result
    
    # def Ki(self, d):
    #     """ Calculate the elastic modulus in a specific direction 
        
    #     Arguments
    #     ---------
    #     d: length 3 array-like
    #         Primary direction

    #     Theory
    #     ------
    #     E(d) = 1/(d ⨂ d : S : d ⨂ d)

    #     Note
    #     ----
    #     The C matrix calculated in this homogenization process is
    #     different than the C matrix defined in the below reference.
        
    #     References
    #     ----------
    #     Nordmann, J., Aßmus, M., & Altenbach, H. (2018). Visualising
    #     elastic anisotropy: theoretical background and computational
    #     implementation. Continuum Mechanics and Thermodynamics, 30(4),
    #     689–708. https://doi.org/10.1007/s00161-018-0635-9 
        
    #     """

    #     assert len(d) == 3, "Input direction vector must be length 3"
    #     d = (np.array(d)/np.linalg.norm(d)).reshape(3)

    #     # Convert to the Voigt notation used in the reference document
    #     CH = fedorovForm(self.CH)
    #     S = np.linalg.inv(CH)

    #     # Calculate the dv vector
    #     d1, d2, d3 = d
    #     dv = np.array([[d1*d1, d2*d2, d3*d3, d1*d2, d2*d3, d3*d1]]).T
    #     dv[3:] *= np.sqrt(2)

    #     return (1/(dv.T @ S @ dv))[0, 0]
    
    # def Kext(self):
    #     """ Calculate the extreme values for the elastic modulus """

    #     # Pull out the compliance tensor
    #     St = self.SHtensor

    #     # Execute jit compiled function for faster run time
    #     outmin, outmax = _Eext(St)

    #     # Export in a more readable format
    #     return dict(max=dict(value=outmax[0], d=outmax[1]), 
    #             min=dict(value=outmin[0], d=outmin[1]))

    #     # return _iext(self.Ei)
        
    # def plotK(self):
    #     """ Plot 3D variation of elastic modulus """

    #     # Create a grid to plot the elastic modulus on
    #     N = 40
    #     M = int(N/2)
    #     PHI, THETA = np.meshgrid(np.linspace(0, np.pi, N),
    #                              np.linspace(0, 2*np.pi, M))
        
    #     # Calculate the elastic modulus at each grid point and map it
    #     # into cartesian space
    #     E = np.zeros((M, N)) 
    #     X = np.zeros((M, N))
    #     Y = np.zeros((M, N))
    #     Z = np.zeros((M, N))
    #     for i in range(M):
    #         for j in range(N):
    #             dij = _d(PHI[i, j], THETA[i, j])
    #             E[i, j] = self.Ei(dij)
    #             X[i, j], Y[i, j], Z[i, j] = E[i, j]*dij
        
    #     # Plot the geometry
    #     surface = go.Surface(x=X, y=Y, z=Z, surfacecolor=E)
    #     fig = go.Figure(data=surface)
    #     fig.update_traces(contours_x=dict(show=True, usecolormap=True,
    #                                     highlightcolor="limegreen", project_x=True),
    #                     contours_y=dict(show=True, usecolormap=True,
    #                                     highlightcolor="limegreen", project_y=True),
    #                     contours_z=dict(show=True, usecolormap=True,
    #                                     highlightcolor="limegreen", project_z=True))
    #     fig.show()

class InternalConductanceHomogenization(InternalHomogenization,
                                        ConductanceHomogenization):
    """ Conductance homogenization based on internal finite element analysis """
    
    def preprocess(self):
        # Aggregate the homogenization data. There are a total of 6 runs
        # that are necessary for the elastic homogenization. Read in
        # each file and pull out the precomputed elasticity
        # coefficients. 

        # Check simulation results
        check = self.check()
        if not check:
            text = ("The homogenization runs failed. Homogenization "
                    "results cannot be processed.")
            logger.error(text)
            raise RuntimeError(text)

        # Read in the results
        with np.load(self._resultfile.with_suffix('.npz')) as data:
            CH = data['CH']
            temps = np.array([data[f'xi{i+1}'] for i in range(self.N)])
            fluxes = np.array([data[f'Bxi{i+1}'] for i in range(self.N)])

        # Store result
        self.CH = CH

        # Store strain data as an nel x 3 x 6 matrix
        self.strains = np.transpose(fluxes, (2, 0, 1))

        # Store temperature data as nnodes x 1 x 6 matrix
        self.displacements = np.transpose(temps, (2, 0, 1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(precision=4)
    mesh = Path(__file__).parent.parent / "mesh/tests/test.npz"
    reuse=False
    # mesh = Path("Database/graph/Diamond/L4_350_W4_350_H4_080_T0_3000/unitcellMesh.npz")
    h = InternalElasticHomogenization(mesh, 1, 0.3)
    # h = InternalConductanceHomogenization(mesh, 1)
    h.run(reuse=reuse)
    h.check()
    h.process(reuse=False, check=False)
    # stress = h.localVMStress([1, 0, 0, 0, 0, 0])
    # stress = h.localVMStress(np.array([[1, 1, 1, 1, 1, 1]]))
    # stress = h.localVMStress(np.array([[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]).T)
    print(h)
    # macroStresses = np.array([
    #     [-0.004049, -0.007476, -0.000172, -3.777e-05, 0.01209, 1.896e-05],
    #     [0.003043, -0.002334, 0.008725, -6.022e-05, 0.008914, 5.436e-05],
    #     [-0.003643, -0.004924, -0.002088, 1.237e-05, 0.01294, -1.071e-05],
    #     [-0.001522, -0.002338, -0.0006459, 2.856e-05, 0.01057, -6.88e-06],
    #     [-0.000468, -0.0008556, -0.0001219, 1.987e-05, 0.00953, 6.911e-06],
    #     [0.005936, 0.002729, 0.01105, 9.042e-06, 0.006727, 5.15e-05],
    #     [0.004901, 0.003373, 0.009675, 0.0001992, 0.005162, -0.0001258],
    #     [0.005073, 0.003433, 0.0102, 9.245e-05, 0.004699, -0.0001412],
    #     [0.01757, 0.01458, 0.02172, -0.0002137, -0.003001, 0.0001582],
    #     [0.01031, 0.004902, 0.01762, -0.0001443, 0.001094, -2.816e-05],
    #     [0.006801, 0.000842, 0.01421, 7.226e-05, 0.002584, -6.572e-05],
    #     [0.006073, 0.00283, 0.01209, -8.076e-05, 0.003797, 2.519e-05],
    #     [0.01294, 0.01294, 0.01392, 1.288e-05, -0.0002015, 8.93e-07],
    #     [0.01308, 0.01312, 0.01416, -5.256e-05, -0.0008898, 1.206e-05],
    #     [0.01695, 0.01607, 0.01937, 1.072e-05, 0.004743, -6.784e-05],
    #     [0.01626, 0.01328, 0.02098, 0.0001025, 0.003037, -1.428e-05],
    #     [0.01289, 0.01285, 0.01399, 6.534e-05, 0.000562, -2.703e-05],
    #     [0.01321, 0.01317, 0.01401, 6.693e-05, 0.00141, -0.000113],
    #     [0.01368, 0.01347, 0.01547, 6.069e-05, 0.002361, -0.0001595],
    #     [0.01365, 0.01366, 0.01461, -5.259e-05, -0.001544, -2.602e-05],
    #     [0.01462, 0.0145, 0.01607, -9.973e-05, -0.002581, -5.39e-05],
    #     [0.01727, 0.0165, 0.01906, -7.641e-05, -0.004462, 0.0001003],
    #     [0.007142, 0.001177, 0.01423, -7.029e-05, -0.003331, -0.000106],
    #     [0.006546, 0.0006463, 0.01354, 0.0002472, -0.004064, 7.574e-05],
    #     [0.007033, 0.0009924, 0.01455, -4.217e-05, -0.002701, -0.0002137],
    #     [0.009217, 0.003668, 0.01677, 7.126e-05, -0.00148, -0.0001881],
    # ]).T





    


