import scipy as sp
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
import pypardiso
import mkl
from scipy.sparse.linalg import cg
# import cvxopt
# import cvxopt.cholmod
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, Array, Value
import ctypes
import numpy as np
import logging
from logging.handlers import QueueHandler, QueueListener
from unitcellengine.mesh.internal import NODE_EL, nodeDofs, periodic
import numpy.typing as npt
import typing
from unitcellengine.utilities import Timing
import unitcellengine.analysis.multiprocess as poolglobals
from unitcellengine.mesh.internal import convert2pyvista
import pyvista as pv
import pyamg

maxThreads = mkl.get_max_threads()
numThreads = 8 if maxThreads > 8 else maxThreads
mkl.set_num_threads(numThreads)

# Create logger
logger = logging.getLogger(__name__)
LOG_LEVEL = logging.DEBUG
BASIC_FORMAT = logging.Formatter("%(levelname)s:%(name)s:%(message)s")


def deleterowcol(A, delrow, delcol):
    """Delete rows and columns from a csc sparse array

    Arguments
    ---------
    A: NxN csc scipy sparse array
        Array to apply row and column deletion
    delrow, delcol: (M,) array-like where M <= N
        Defines the row and column numbers to delete from *A*

    Returns
    -------
    A: (N-M)x(N-M) csc sparse array
        Initial array with correspnding rows and columns removed

    """
    # Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A


def isotropicC(E, nu):
    """Create isotropic stiffness matrix in Voigt notations

    Arguments
    ---------
    E: float > 0
        Elastic modulus of the material
    nu: -1.5 < float < 0.5
        Poisson's ration for the material

    Returns
    -------
    C: 6x6 numpy array
        Elastic stiffness matrix in Voigt notations (formatted for
        engineering strains rather than tensorial strains)

    """
    return (
        E
        / ((1 + nu) * (1 - 2 * nu))
        * np.array(
            [
                [1 - nu, nu, nu, 0, 0, 0],
                [nu, 1 - nu, nu, 0, 0, 0],
                [nu, nu, 1 - nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
            ]
        )
    )


def isotropicK(k: float) -> npt.ArrayLike:
    """Create isotropic conductance matrix

    Arguments
    ---------
    k: float > 0
        Material conductance

    Returns
    -------
    C: 3x3 numpy array
        Conductance matrix

    """
    return np.eye(3) * k


def ke(
    C: npt.ArrayLike = isotropicC(1, 0.3), L: float = 1, W: float = 1, H: float = 1
) -> typing.Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """Element stiffness matrix for 8 node 'rectangular' hex element

    Keywords
    --------
    C: NxN numpy array, where N=3 or 6
        Elastic stiffness matrix (in Voigt notiation for engineer
        strains) or conductance matrix.
    L, W, H: float > 0
        Length, width, and height of the element

    Returns
    -------
    ke: (8*dof/node, 8*dof/node) numpy array
        Element matrix
    F: (8*dof/node, N) numpy array
        Homogenization body force matrix for each unit case
    B: (8, N, 24) numpy array
        Element derivative matrix for each integration point
    detJ: (8,) numpy array
        Determinant of the Jacobian matrix at each integration point

    Notes
    -----
    - For elastic problems, dof/node = 3
    - For conductance problems, dof/node = 1
    - This element is integrated with Gauss Quadrature using 8
      integration points that *all have weights of 1*.
    - This is an element centered about a (0, 0, 0) origin
    - This element is only valid for rectangular elements
    """

    # Determine the element physics type (stiffness or conductance)
    N = C.shape[0]
    assert C.shape[0] == C.shape[1], "Material matrix must be square."
    assert N == 3 or N == 6, "Material matrix must be either 3x3 or 6x6."

    if N == 3:
        DOF_NODE = 1
    else:
        DOF_NODE = 3

    #
    # Gauss points coordinates on each direction
    # NOTE: the weights for these gauss points are all 1
    gaussPoints = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    #
    # Matrix of vertices coordinates. Generic element centred at the origin.
    coordinates = np.zeros((8, 3))
    coordinates[0, :] = [-L / 2, -W / 2, -H / 2]
    coordinates[1, :] = [L / 2, -W / 2, -H / 2]
    coordinates[2, :] = [L / 2, W / 2, -H / 2]
    coordinates[3, :] = [-L / 2, W / 2, -H / 2]
    coordinates[4, :] = [-L / 2, -W / 2, H / 2]
    coordinates[5, :] = [L / 2, -W / 2, H / 2]
    coordinates[6, :] = [L / 2, W / 2, H / 2]
    coordinates[7, :] = [-L / 2, W / 2, H / 2]
    #
    # Preallocate memory for stiffness matrix
    K = np.zeros((8 * DOF_NODE, 8 * DOF_NODE))
    F = np.zeros((8 * DOF_NODE, N))
    # Loop over each Gauss point
    B = np.zeros((8, N, 8 * DOF_NODE))
    detJ = np.zeros(8)
    k = 0
    for xi1 in gaussPoints:
        for xi2 in gaussPoints:
            for xi3 in gaussPoints:
                # Compute shape functions derivatives
                dShape = (1 / 8) * np.array(
                    [
                        [
                            -(1 - xi2) * (1 - xi3),
                            (1 - xi2) * (1 - xi3),
                            (1 + xi2) * (1 - xi3),
                            -(1 + xi2) * (1 - xi3),
                            -(1 - xi2) * (1 + xi3),
                            (1 - xi2) * (1 + xi3),
                            (1 + xi2) * (1 + xi3),
                            -(1 + xi2) * (1 + xi3),
                        ],
                        [
                            -(1 - xi1) * (1 - xi3),
                            -(1 + xi1) * (1 - xi3),
                            (1 + xi1) * (1 - xi3),
                            (1 - xi1) * (1 - xi3),
                            -(1 - xi1) * (1 + xi3),
                            -(1 + xi1) * (1 + xi3),
                            (1 + xi1) * (1 + xi3),
                            (1 - xi1) * (1 + xi3),
                        ],
                        [
                            -(1 - xi1) * (1 - xi2),
                            -(1 + xi1) * (1 - xi2),
                            -(1 + xi1) * (1 + xi2),
                            -(1 - xi1) * (1 + xi2),
                            (1 - xi1) * (1 - xi2),
                            (1 + xi1) * (1 - xi2),
                            (1 + xi1) * (1 + xi2),
                            (1 - xi1) * (1 + xi2),
                        ],
                    ]
                )

                # Compute Jacobian matrix
                J = dShape @ coordinates
                # Compute auxiliar matrix for construction of B-Operator
                auxiliar = np.linalg.inv(J) @ dShape

                # Preallocate memory for B-Operator
                # [dN/dx  0     0    ] epsx
                # [0      dN/dy 0    ] espy
                # [0      0     dN/dz] epsz
                # When shear derivatives are needed, add the following
                # [dN/dy  dN/dx 0    ] gammaxy
                # [0      dN/dz dN/dy] gammayz
                # [dN/dz  0     dN/dx] gammaxz

                if N == 3:
                    for i in range(3):
                        for j in range(8):
                            B[k, i, j] = auxiliar[i, j]
                else:
                    # Construct first three rows
                    for i in range(3):
                        for j in range(8):
                            B[k, i, 3 * j + i] = auxiliar[i, j]

                    # Include shear derivatives
                    # Construct fourth row
                    for j in range(8):
                        B[k, 3, 3 * j + 0] = auxiliar[1, j]

                    for j in range(8):
                        B[k, 3, 3 * j + 1] = auxiliar[0, j]

                    # Construct fifth row
                    for j in range(8):
                        B[k, 4, 3 * j + 1] = auxiliar[2, j]

                    for j in range(8):
                        B[k, 4, 3 * j + 2] = auxiliar[1, j]

                    # Construct sixth row
                    for j in range(8):
                        B[k, 5, 3 * j + 0] = auxiliar[2, j]

                    for j in range(8):
                        B[k, 5, 3 * j + 2] = auxiliar[0, j]

                # Add to stiffness matrix
                # NOTE: the gauge point weight isn't present here because
                #       it is 1 for this forumlation
                detJ[k] = np.linalg.det(J)
                f = (B[k, :, :].T @ C) * detJ[k]
                K += f @ B[k, :, :]
                # (B.T @ C @ B)*np.linalg.det(J)
                # F += (B.T @ C)*np.linalg.det(J)
                F += f
                k += 1
    return K, F, B, detJ


def _initParallelSolve(sK, iK, jK, F, B, cases, ndof, rtol, atol, queue):
    """Initialize shared memory variables for parallel linear solves"""
    # Store shared arrays
    poolglobals.sK = sK
    poolglobals.iK = iK
    poolglobals.jK = jK
    poolglobals.F = F
    poolglobals.B = B
    poolglobals.cases = cases
    poolglobals.rtol = rtol
    poolglobals.atol = atol
    poolglobals.ndof = ndof

    # Setup logging
    h = QueueHandler(queue)
    logger = logging.getLogger(__name__)
    logger.addHandler(h)
    logger.setLevel(LOG_LEVEL)
    poolglobals.logger = logger


def _isolve(i):
    """Iterative solver function used when solving in parallel"""
    ndof = poolglobals.ndof.value
    cases = poolglobals.cases.value
    # Create a csc_matrix with the copy=False option so as to just read
    # the shared memory.
    K = csr_matrix(
        (poolglobals.sK, (poolglobals.iK, poolglobals.jK)),
        shape=(ndof, ndof),
        copy=False,
    )
    # Use np.frombuffer to interpret data from share memory rather than
    # creating a new object in memory
    F = np.frombuffer(poolglobals.F).reshape((ndof, cases), order="F")
    B = np.frombuffer(poolglobals.B).reshape((ndof, cases), order="F")
    rtol = poolglobals.rtol.value
    atol = poolglobals.atol.value
    logger = poolglobals.logger

    # Define preconditioner
    # @TODO This is kind of an expensive process that can be shared
    # across all processes. So, it should probably be precomputed and
    # then shared rather than computed within each process
    # M = sp.sparse.diags(1/K.diagonal()) # Jacobi preconditioner for iterative solver
    # M = ilupp.IChol0Preconditioner(K.tocsc())
    # Monkey patch for Windows which doesn't support float128
    # See https://github.com/pyamg/pyamg/issues/273
    if not hasattr(np, "float128"):
        np.float128 = np.longdouble  # #698
    M = pyamg.smoothed_aggregation_solver(K, B=B).aspreconditioner()

    # Setup iteration history callback function
    it = 0

    def callback(xk):
        nonlocal it
        if it % 20 == 0:
            logger.debug(f"Case {i}/Iteration {it}: {np.linalg.norm(K@xk-F[:,i])}")
        it += 1

    return cg(K, F[:, i], M=M, tol=rtol, atol=atol, callback=callback)


# def elasticHomogenization(mesh, E, nu, filename=None,
#                           solver="auto", parallel=True,
#                           rtol=1e-6, atol=1e-8,):

#     C = isotropicC(E, nu)
#     return homogenization(mesh, C, filename=None,
#                           solver="auto", parallel=True,
#                           rtol=1e-6, atol=1e-8,)


def homogenization(
    mesh,
    C,
    filename=None,
    solver="auto",
    parallel=True,
    rtol=1e-6,
    atol=1e-8,
):
    """Solve homogenization equations for a given mesh

    Arguments
    ---------
    mesh: str or Path
        Defines the .npz mesh filename as generated by
        unitcell.mesh.internal (which corresponds to an ersatz material
        based representation of a geometry over a voxel mesh.)
    C: 3x3 or 6x6 numpy array
        Elastic or conductance material matrix for the base material.
        Use *isotropicC* and *isotropicK* helper functions to define
        these matrices. In the case of elastic homogenization, C must be
        in engineering form (i.e., for use with engineering shear strain
        rather than tensorial.)

    Keywords
    --------
    filename: None or str or Path (default is None)
        If str or Path, defines the filename to save the results to.
        The name must have an extension of either .npz or .vtu. If no
        file extension exists, then both .npz and .vtu representations
        will be saved.
    solver: "direct", "iterative" or "auto" (default = "auto")
        Linear solver to use to solve finite element problem
    parallel: boolean or int > 0 (default is True)
        When using the "iterative" *solver", run all processes in
        parallel if True, don't run in parallel if false, and use
        *parallel* processors if an integer. Note, there are no benefits
        to using more than the number of subcases in the homogenization
        (for example, for elastic homogenization, there is no benefit to
        using more than 6 processes).
    rtol: float > 0 (default = 1e-6)
        Relative convergence tolerance for the iterative solver. This
        keyword is only relevant when *solver*="iterative" or "auto".
    atol: float > 0 (default = 1e-8)
        Absolute convergence tolerance for the iterative solver. This
        keyword is only relevant when *solver*="iterative" or "auto".

    Returns
    -------
    CH: 3x3 or 6x6 numpy array
        The homogenized matrix for the input mesh.

    """

    # Determine the element physics type (stiffness or conductance)
    N = C.shape[0]
    assert C.shape[0] == C.shape[1], "Material matrix must be square."
    assert N == 3 or N == 6, "Material matrix must be either 3x3 or 6x6."
    if N == 3:
        DOF_NODE = 1
        CASES = 3
        logger.info("Solving conductance homogenization.")
    else:
        DOF_NODE = 3
        CASES = 6
        logger.info("Solving elastic homogenization.")
    DOF_EL = NODE_EL * DOF_NODE

    # Load mesh
    mesh = Path(mesh)
    with np.load(mesh) as data:
        ns = data["nodes"]
        nsdf = data["nsdf"]
        ninds = data["ninds"]
        e2n = data["e2n"]
        (nelx, nely, nelz) = data["nels"]
        rhos = data["erhos"]
        einds = data["einds"]

    # Overall mesh properties
    ndof = ns.shape[0] * DOF_NODE  # Total number of degrees of freedom
    dofs = np.arange(ndof)  # DOF vector

    # Make the mesh periodic
    pe2n, lns, fns = periodic(ns, e2n, [nelx, nely, nelz])
    ldofs = nodeDofs(lns, DOF_NODE)
    fdofs = nodeDofs(fns, DOF_NODE)

    # Define inactive nodes (i.e., nodes that aren't in any elements
    # a positive density). Note, the initial active node list created by
    # the meshing operation doesn't take into account periodicity. So,
    # special care needs to be taken when defining these nodes.
    alns = np.intersect1d(ns[ninds, 0], lns)  # Active leader nodes
    _, _, afinds = np.intersect1d(
        ns[ninds, 0], fns, return_indices=True
    )  # Active follower nodes
    keep = np.setdiff1d(lns[afinds], alns)
    empty = nodeDofs(np.setdiff1d(ns[~ninds, 0], keep), DOF_NODE).flatten("F")

    # Section dofs
    pin = nodeDofs(alns[0], DOF_NODE).flatten("F")
    remove = np.unique(np.hstack((empty, fdofs.flatten("F"), pin))).astype(np.int64)
    free = np.setdiff1d(dofs, remove).astype(np.int64)

    # Pull out the active mesh properties
    subrhos = rhos[einds]
    # e2dof = nodeDofs(e2n.flatten(), DOF_NODE).flatten('F').reshape(-1, DOF_EL)
    pe2dof = nodeDofs(pe2n.flatten(), DOF_NODE).flatten("F").reshape(-1, DOF_EL)
    # sube2dof = e2dof[einds, :]
    subpe2dof = pe2dof[einds, :]
    # subnel = sube2dof.shape[0]

    # Construct element elastic matrices
    L, W, H = ns[:, 1:].max(axis=0) - ns[:, 1:].min(axis=0)
    eL, eW, eH = L / nelx, W / nely, H / nelz
    KE, FE, BEs, detJs = ke(C=C, L=eL, W=eW, H=eH)

    # Assemble the full stiffness matrix
    # Here, the elements with partial density are scaled linearly
    # according to the density
    with Timing("Assemble stiffness matrix", logger):
        iK = np.tile(subpe2dof, DOF_EL).flatten()
        jK = np.tile(subpe2dof.flatten(), (DOF_EL, 1)).flatten("F")
        sK = (
            (KE.flatten("F")[np.newaxis]).T * (1e-4 + (subrhos) * (1 - 1e-4))
        ).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = deleterowcol(K, remove, remove)

    # Assemble homogenization load cases
    # Here, the elements with partial density are scaled linearly
    # according to the density
    with Timing("Assembly homogenization body loads", logger):
        sF = (
            (FE.flatten("F")[np.newaxis]).T * (1e-4 + (subrhos) * (1 - 1e-4))
        ).flatten(order="F")
        iF = np.tile(subpe2dof, CASES).flatten()
        jF = np.tile(
            np.arange(CASES), (subpe2dof.shape[1], subpe2dof.shape[0])
        ).flatten("F")
        F = coo_matrix((sF, (iF, jF)), shape=(ndof, CASES)).toarray()

    # Solve the system of equations
    u = np.zeros((ndof, CASES))

    subndof = free.shape[0]
    logger.debug(
        f"There are {subndof} degrees of freedom in the "
        f"the {einds.sum()} element problem."
    )
    if "auto" in solver.lower():
        # Determine the appropriate solver based on the DOF count in the
        # stiffness matrix
        # Ref: 769227 DOF required about 36 GB of RAM and 3 min solve
        #      time with the direct solver while 5 GB of RAM and 10.5 min
        #      solve time with the CG solver
        if subndof > 750e3:
            solver = "iterative"
            logger.debug(
                "There are a large number of degrees of "
                "freedom in this problem. Solving with the "
                "iterative solver."
            )
        else:
            solver = "direct"
            logger.debug(
                "There are a modest number of degrees of "
                "freedom in this problem. Solving with the "
                "direct solver."
            )

    if "direct" in solver.lower():
        with Timing("Solving homogenization with a direct solver", logger):
            # # Solve system using a direct solver
            pardiso = pypardiso.PyPardisoSolver(mtype=2) # Real/symmetric/positive definite
            # pardiso.set_statistical_info_on()
            u[free, :] = pardiso.solve(sp.sparse.triu(K, format="csr"), F[free, :])            

            # Kr = cpx.scipy.sparse.coo_matrix(K)
            # Fr = cpx.scipy.sparse.coo_matrix(F)
            # ur = cpx.scipy.sparse.spsolve(Kr, Fr)
            # Kr = cvxopt.spmatrix(K.data, K.row.astype(np.int64), K.col.astype(np.int64))
            # Fr = cvxopt.matrix(F[free, :])
            # cvxopt.cholmod.linsolve(Kr, Fr)
            # u[free, :] = np.array(Fr)
    else:
        # Compute rigid body modes, which are used as input to the
        # interative solver to speed up convergence.
        if DOF_NODE == 3:
            B = np.zeros((ndof, 6))
            B[0::3, 0] = 1  # vector field in x direction
            B[1::3, 1] = 1  # vector field in y direction
            B[2::3, 2] = 1  # vector field in z direction

            B[0::3, 3] = -ns[:, 2]  # rotation vector field (-y, x, 0)
            B[1::3, 3] = ns[:, 1]
            B[0::3, 4] = -ns[:, 3]  # rotation vector field (-z, 0, x)
            B[2::3, 4] = ns[:, 1]
            B[1::3, 5] = -ns[:, 3]  # rotation vector field (0,-z, y)
            B[2::3, 5] = ns[:, 2]

            B = B[free, :]
        else:
            B = np.ones((len(free), 1))

        with Timing(
            "Solving homogenization equations with an iterative solver", logger=logger
        ):
            # Determine if the jobs should be run in serial or parallel
            if parallel:
                if isinstance(parallel, bool):
                    pools = 3
                else:
                    pools = parallel
                logger.debug(f"Parallel run with {pools} processors.")

                # Create shared memory pointers for all of the data required to
                # each linear system of equations. Note: this is required to
                # prevent process locking. If this isn't implemented, each process
                # essentially runs sequentially without any speed up. See
                # https://stackoverflow.com/questions/1675766/combine-pool-map-with-shared-memory-array-in-python-multiprocessing
                # Note: only preallocate the memory here, rather than
                # loading the data in (ex: Array(ctypes.c_longdouble,
                # K.data, lock=False)) as this is incredibly slow because
                # the data is copied over element by element. You can load
                # the data is a faster way later.
                sK = Array(ctypes.c_longdouble, K.data.shape[0], lock=False)
                iK = Array(ctypes.c_int64, K.row.shape[0], lock=False)
                jK = Array(ctypes.c_int64, K.col.shape[0], lock=False)
                sF = Array(
                    ctypes.c_longdouble, F[free, :].flatten("F").shape[0], lock=False
                )
                sB = Array(ctypes.c_longdouble, B.flatten("F").shape[0], lock=False)
                srtol = Value("f", lock=False)
                satol = Value("f", lock=False)
                sndof = Value(ctypes.c_int64, lock=False)
                cases = Value(ctypes.c_int16, lock=False)

                # Setup multiprocess safe logging using a queue and listener
                queue = multiprocessing.Queue(-1)
                handler = logging.StreamHandler()
                handler.setFormatter(BASIC_FORMAT)
                listener = QueueListener(queue, handler)
                listener.start()
                mlogger = multiprocessing.log_to_stderr()
                mlogger.setLevel(logging.INFO)

                # Create a pool of works to solve all cases
                # Note, the global shared variables need to be initialized
                # for each process, requiring the intializer and initargs.
                with Pool(
                    pools,
                    initializer=_initParallelSolve,
                    initargs=(sK, iK, jK, sF, sB, cases, sndof, srtol, satol, queue),
                ) as p:
                    # Initialize the share variables after the pool had been
                    # generated. If this isn't done after the pool is
                    # started, there won't be any values in the global
                    # variables. Note, you can copy the data with sK[:] =
                    # K.data, but this operation is rather slow. A much
                    # better way is as done below, using np.ctypeslib.as_array.
                    # https://newbedev.com/why-are-multiprocessing-sharedctypes-assignments-so-slow
                    np.ctypeslib.as_array(sK)[:] = K.data
                    np.ctypeslib.as_array(iK)[:] = K.row
                    np.ctypeslib.as_array(jK)[:] = K.col
                    np.ctypeslib.as_array(sF)[:] = F[free, :].flatten("F")
                    np.ctypeslib.as_array(sB)[:] = B.flatten("F")
                    srtol.value = rtol
                    satol.value = atol
                    sndof.value = K.get_shape()[0]
                    cases.value = CASES

                    # Run the solution cases. Note, order matters here, so
                    # don't use any of the "_unordered" mappings such as
                    # "imap_unordered" because the displacement arrays can
                    # be stored in the incorrect column, leading to
                    # incorrect results.
                    for i, (u0, ecode) in enumerate(p.map(_isolve, range(CASES))):
                        if ecode == 0:
                            logger.debug(f"Homogenization case {i} completed.")
                            u[free, i] = u0
                        else:
                            raise RuntimeError(
                                f"Iterative solver run {i} failed to converge: {ecode}"
                            )
                listener.stop()

            else:
                # Serial run
                logger.debug("Serial run")

                # Monkey patch for Windows which doesn't support float128
                # See https://github.com/pyamg/pyamg/issues/273
                if not hasattr(np, "float128"):
                    np.float128 = np.longdouble  # #698

                # Generate multigrid preconditioner. This has been found
                # to be much more efficient than Jacobi and iLU
                # preconditioners.
                ml = pyamg.smoothed_aggregation_solver(K.tocsr(), B=B)
                M = ml.aspreconditioner()

                for i in range(CASES):
                    # Setup iteration history callback function
                    it = 0

                    def callback(xk):
                        nonlocal it
                        if it % 20 == 0:
                            logger.debug(
                                f"Case {i}/Iteration {it}: {np.linalg.norm(K@xk-F[free,i])}"
                            )
                        it += 1

                    with Timing(
                        f"Solving homogenization case {i+1} of {CASES}", logger
                    ):
                        # Solve using a Conjugate Gradient iterative solver
                        u0, ecode = cg(
                            K, F[free, i], M=M, tol=rtol, atol=atol, callback=callback
                        )
                    if ecode == 0:
                        # If successful, store the solution array
                        u[free, i] = u0
                    else:
                        text = f"Iterative solver run {i+1} failed to converge: {ecode}"
                        logger.error(text)
                        raise RuntimeError(text)

    # For the "follower" solutions solution to be periodic according to
    # "leaders"
    u[fdofs.flatten("F"), :] = u[ldofs.flatten("F"), :]
    F[fdofs.flatten("F"), :] = F[ldofs.flatten("F"), :]

    # Calculate homogenized properties
    # Calculate local amplification matrix for all elements and gauss points
    # CH = sum(Vel/V*C[I - B xi]) = sum(Vel/V*ρ_el*C_elastic[I - B xi])
    # Note: The elastic stiffness needs to be scaled based on the element density
    V = L * W * H
    ax = np.newaxis
    Bxi = np.tensordot(BEs, u[subpe2dof, :], (2, 1))  # 8 x N x nel x N
    eampe = np.eye(CASES)[ax, ax].transpose((0, 2, 1, 3)) - Bxi  # 8 x N x nel x N
    # sampe = (np.tensordot(C, eampe, (1, 1))*rhos[einds][ax, ax, ax].transpose((0, 1, 3, 2))).transpose((1, 0, 2, 3))
    sampe = np.tensordot(C, eampe, (1, 1)).transpose((1, 0, 2, 3))  # 8 x N x nel x N
    scaling = (detJs[ax].T * rhos[einds][ax])[ax, ax].transpose((2, 0, 3, 1))
    CH = (scaling * sampe).sum(axis=0).sum(axis=1) / V

    # Save results
    if filename:
        filename = Path(filename)

        # Caclulate centroid strains (gauss point average)
        dxis = Bxi.sum(axis=0) / Bxi.shape[0]  # N x nel x N
        # ceampe = eampe.sum(axis=0)/sampe.shape[0]
        # csampe = sampe.sum(axis=0)/sampe.shape[0]
        # ceampe = Bxi.sum(axis=0)/Bxi.shape[0]
        # tmp = np.tensordot(C, Bxi, (1, 1)).transpose((1, 0, 2, 3))
        # tmp = (np.tensordot(C, Bxi, (1, 1))*rhos[einds][ax, ax, ax].transpose((0, 1, 3, 2))).transpose((1, 0, 2, 3))
        # csampe = tmp.sum(axis=0)/tmp.shape[0]
        # ceampe = eampe.max(axis=0)
        # csampe = sampe.max(axis=0)

        # Save as a paraview file
        pointData = {"sdf": nsdf}
        cellData = {"density": subrhos}
        # pointData['scalars'] =
        # cellData['scalars'] =
        for i in range(CASES):
            # Note: ascontiguousarray is required by the unstructuredGridToVTK export
            us = np.ascontiguousarray(u[:, i].reshape((-1, DOF_NODE)).T)
            pointData[f"xi{i+1}"] = tuple(us)

            # Fx, Fy, Fz = np.ascontiguousarray(F[:, i].reshape((-1, 3)).T)
            # pointData[f"force{i+1}"] = (Fx, Fy, Fz)

            dus = np.ascontiguousarray(dxis[i, :, :].T)
            cellData[f"Bxi{i+1}"] = tuple(dus)

            # exx, eyy, ezz, exy, eyz, exz = np.ascontiguousarray(ceampe[i, :, :].T)
            # cellData[f'strainAmp{i+1}'] = (exx, eyy, ezz, exy, eyz, exz)

            # sxx, syy, szz, sxy, syz, sxz = np.ascontiguousarray(csampe[i, :, :].T)
            # cellData[f'stressAmp{i+1}'] = (sxx, syy, szz, sxy, syz, sxz)

        # Parse the input suffix
        suffix = filename.suffix
        if suffix == "":
            suffixes = [".npz", ".vtu"]
        elif not (suffix == ".npz" or suffix == ".vtu"):
            logger.warning(
                f"Suffix {suffix} is an unsupported export "
                "Exporting as '.npz' and '.vtu' instead."
            )
            suffixes = [".npz", ".vtu"]
        else:
            suffixes = [suffix]

        # Write out files for each file type specified
        for suffix in suffixes:
            if suffix == ".npz":
                # Save as a python numpy array
                save = {k: v for k, v in {**pointData, **cellData}.items()}
                save["C"] = C
                save["CH"] = CH
                save["B"] = BEs
                save["Ve"] = eL * eW * eH
                save["V"] = L * W * H
                with Timing(
                    f"Saving result to {filename.with_suffix('.npz')}", logger=logger
                ):
                    np.savez_compressed(filename.with_suffix(".npz"), **save)
            else:
                with Timing(
                    f"Saving result to {filename.with_suffix('.vtu')}", logger=logger
                ):
                    grid = convert2pyvista(mesh, pointData=pointData, cellData=cellData)
                    grid.save(filename_with(suffix(".vtu")))

    return CH


def internal2vtu(mesh, result, filename=None):
    """Convert internal result file to Paraview compatabile file

    Arguments
    ---------
    mesh: str or Path object with .npz extension
        Filename of mesh file generated by internal homogenization
        engine.
    result: str or Path object with .npz extension
        Filename of result file generated by the internal homogenization
        engine.

    Keywords
    ---------
    filename: None or str or Path object with .vtu extension
        Filename of vtu file. If None, the same basename as *result* is
        used.

    """

    # Process result filename
    result = Path(result)
    assert result.suffix == ".npz", (
        f"Input result {result} has the " "incorrect suffix. Should be .npz."
    )

    pointData = {}
    cellData = {}

    # Load results
    with np.load(result) as data:
        SH = np.linalg.inv(data['CH'])
        C = data['C']
        xis = {file: data[file].T for file in data.files if file[:2] == "xi"}
        Bxis = {file: data[file].T for file in data.files if file[:3] == "Bxi"}
    pointData.update(xis)
    cellData.update(Bxis)

    # Create the pyvista UnstructuredGrid object and save it
    grid = convert2pyvista(mesh, pointData=pointData, cellData=cellData)

    filename = Path(filename) if filename else result.with_suffix(".vtu")

    grid.save(filename.with_suffix(".vtu"))



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Load mesh
    # mesh = Path("unitcell/mesh/tests/test.npz")
    # C = isotropicC(1, 0.35)
    mesh = Path("Database/graph/Diamond/L4_350_W4_350_H4_080_T0_3000/unitcellMesh.npz")
    C = isotropicC(1, 0.3)

    # CH = homogenization(mesh, C, Path("test.npz"))

    # # C = isotropicK(1)
    # # CH1 = homogenization(mesh, C,
    # #                     Path("unitcell/analysis/tests/test.npz"),
    # #                     solver="direct")
    # # CH2 = homogenization(mesh, C,
    # #                     Path("unitcell/analysis/tests/test.npz"),
    # #                     solver="iterative")
    # CH3 = homogenization(
    #     mesh,
    #     C,
    #     Path("unitcell/analysis/tests/test.npz"),
    #     solver="iterative",
    #     parallel=False,
    # )
    #
    # np.set_printoptions(precision=3)
    # # print(CH1)
    # # print(CH2)
    # print(CH3)

    # internal2vtu(mesh, Path("unitcell/analysis/tests/test.npz"))
    internal2vtu(mesh, Path("test.npz"))
    # with np.load(filename) as data:
    #     ns = data['nodes']
    #     nsdf = data['nsdf']
    #     ninds = data['ninds']
    #     e2n = data['e2n']
    #     (nelx, nely, nelz) = data['nels']
    #     rhos = data['erhos']
    #     einds = data['einds']

    # # ns, nsdf, ninds, e2n, (nelx, nely, nelz), rhos, einds = mesh(geometry.sdf, elementSize=elementSize, dimensions=dims, filename=filename)
    # # ns, nrhos, ninds, e2n, e2dof, rhosmean, einds = mesh(box(dims), elementSize=elementSize, dimensions=dims)
    # ndof = ns.shape[0]*DOF_NODE
    # dofs = np.arange(ndof)

    # # Make element mappings periodic
    # # NOTE: This relies on the known structured definition of the nodes
    # Ns = ns[:, 0].reshape((nelx+1, nely+1, nelz+1), order='F').astype(np.int64)
    # lns = np.hstack((Ns[0, :, :].flatten(),
    #                    Ns[:, 0, :].flatten(),
    #                    Ns[:, :, 0].flatten()))
    # ldofs = nodeDofs(lns)
    # fns = np.hstack((Ns[-1, :, :].flatten(),
    #                    Ns[:, -1, :].flatten(),
    #                    Ns[:, :, -1].flatten()))
    # fdofs = nodeDofs(fns)
    # Ns[-1, :, :] = Ns[0, :, :]
    # Ns[:, -1, :] = Ns[:, 0, :]
    # Ns[:, :, -1] = Ns[:, :, 0]
    # pns = Ns.flatten('F')
    # pe2n = pns[e2n]

    # subrhos = rhos[einds]
    # e2dof = nodeDofs(e2n.flatten(), DOF_NODE).flatten('F').reshape(-1, DOF_EL)
    # pe2dof = nodeDofs(pe2n.flatten(), DOF_NODE).flatten('F').reshape(-1, DOF_EL)
    # # e2dof[el, :] = nodeDofs(e2n[el, :]).T.flatten()
    # sube2dof = e2dof[einds, :]
    # subpe2dof = pe2dof[einds, :]
    # subnel = sube2dof.shape[0]

    # # Construct the index pointers for the mesh
    # # NOTE: switched from kron to something faster
    # # iK = np.kron(sube2dof,np.ones((DOF_EL, 1))).flatten()
    # # jK = np.kron(sube2dof,np.ones((1, DOF_EL))).flatten()

    # # Construct element vectors
    # L, W, H = ns[:, 1:].max(axis=0) - ns[:, 1:].min(axis=0)
    # KE, FE, BEs, detJs = ke(C=isotropicC(1, 0.3), L=L/nelx, W=W/nely, H=H/nelz)

    # # Assemble full stiffness matrix
    # iK = np.tile(subpe2dof, DOF_EL).flatten()
    # jK = np.tile(subpe2dof.flatten(), (DOF_EL, 1)).flatten('F')
    # sK=((KE.flatten('F')[np.newaxis]).T*(1e-4+(subrhos)*(1-1e-4))).flatten(order='F')
    # K = coo_matrix((sK,(iK,jK)), shape=(ndof,ndof)).tocsc()

    # # Section dofs
    # empty = nodeDofs(ns[~ninds, 0]).flatten('F')
    # # fixedxy = np.array([0, 1,])
    # # fixedy =  nodeDofs(ns[np.logical_and(np.logical_and(ns[:, -1]<-0.49, ns[:, 1] > 0.49), ns[:, 2] < -0.49), 0])[1, :]
    # # fixedz = nodeDofs(ns[ns[:, -1] < -0.49, 0])[-1, :]
    # # fixed = np.hstack((fixedxy, fixedz, fixedy))
    # # fixed = np.array([0, 1, 2])
    # fixed = np.array([])
    # remove = np.unique(np.hstack((empty, fixed, fdofs.flatten('F')))).astype(np.int64)
    # free = np.setdiff1d(dofs, remove).astype(np.int64)

    # # Pin a node to remove rigid body modes
    # remove = np.hstack((remove, free[:3]))
    # free = free[3:]

    # # Assemble homogenization load cases
    # sF = ((FE.flatten('F')[np.newaxis]).T*(1e-4+(subrhos)*(1-1e-4))).flatten(order='F')
    # # sF = np.tile(FE.flatten(), subnel)
    # # IF, JF = np.meshgrid(sube2dof.flatten(), np.arange(6))
    # iF = np.tile(subpe2dof, 6).flatten()
    # jF = np.tile(np.arange(6), (subpe2dof.shape[1], subpe2dof.shape[0]) ).flatten('F')
    # F = coo_matrix((sF, (iF,jF)), shape=(ndof, 6)).toarray()

    # # # Create unit load on top surface
    # # F = np.zeros(ndof)
    # # fsubdofs = nodeDofs(ns[ns[:, -1] > 0.49, 0])[2, :]
    # # F[fsubdofs] = 1./fsubdofs.shape[0]

    # # Solve system
    # K = deleterowcol(K, remove, remove).tocoo()
    # # Kr = cvxopt.spmatrix(K.data,K.row.astype(np.int64),K.col.astype(np.int64))
    # # # # B = cvxopt.matrix(F[free])
    # # Br = cvxopt.matrix(F[free, :])
    # # cvxopt.cholmod.linsolve(Kr, Br)
    # # uref = np.zeros((ndof, 6))
    # # uref[free, :] = np.array(Br)
    # # u = np.zeros(ndof)
    # u = np.zeros((ndof, 6))
    # M = sp.sparse.diags(1/K.diagonal()) # Jacobi preconditioner

    # # with Pool(6) as p:
    # #     cases = [(K, F[free, i], M) for i in range(6)]
    # #     func = partial(isolve, K=K, F=F[free, :], M=M)
    # #     for i, (u0, ecode) in enumerate(p.imap_unordered(func, range(6))):
    # #     # for i, (u0, ecode) in enumerate(p.imap_unordered(isolve,
    # #     #                                 zip(itertools.repeat(K),
    # #     #                                        [F[free, i] for i in range(6)],
    # #     #                                        itertools.repeat(K)))):
    # #     # for i, (u0, ecode) in enumerate(p.imap_unordered(partial(isolve, K=K, F=F[free, :], M=M),
    # #     #                                                  range(6))):
    # #         if ecode == 0:
    # #             u[free, i] = u0
    # #         else:
    # #             raise RuntimeError(f"Iterative solver run {i} failed to converge: {ecode}")

    # for i in range(F.shape[1]):
    #     u0, ecode = cg(K, F[free, i], M=M, tol=1e-6, atol=1e-8)
    #     if ecode == 0:
    #         u[free, i] = u0
    #     else:
    #         raise RuntimeError(f"Iterative solver run {i} failed to converge: {ecode}")
    # # u[free] = np.array(B)[:,0]
    # # u[free, :] = np.array(B)
    # u[fdofs.flatten('F'), :] = u[ldofs.flatten('F'), :]

    # # Calculate homogenized properties
    # C = isotropicC(1, 0.3)
    # # Calculate local amplification matrix for all elements and gauss points
    # # CH = sum(Vel/V*C[I - B xi])
    # V = L*W*H
    # ax = np.newaxis
    # Bxi = np.tensordot(BEs, u[subpe2dof, :], (2, 1)) # 8 x 6 x nel x 6
    # eampe = np.eye(6)[ax, ax].transpose((0, 2, 1, 3))-Bxi # 8 x 6 x nel x 6
    # sampe = np.tensordot(C, eampe, (1, 1))*rhos[einds][ax, ax, ax].transpose((0, 1, 3, 2))
    # CH = (detJs[ax, ax, ax].transpose((0, 3, 1, 2))*sampe).sum(axis=1).sum(axis=1)/V
    # CH += CH.T
    # CH /= 2

    # # Save results
    # directory=Path(__file__).parent/Path("tests")
    # nel = subrhos.shape[0]
    # for i in range(6):
    #     ux, uy, uz = np.ascontiguousarray(u[:, i].reshape((-1, 3)).T)
    #     unstructuredGridToVTK(str(directory / Path(f'test{i}')),
    #                           ns[:, 1], ns[:, 2], ns[:, 3],
    #                           e2n[einds, :].flatten().astype(np.int64),
    #                           np.ravel_multi_index((np.arange(nel), [7]), (nel, 8))+1,
    #                           np.array([VtkHexahedron.tid]*nel),
    #                           cellData={"density": subrhos},
    #                           pointData={"sdf": nsdf,
    #                                      "displacement": (ux, uy, uz)
    #                                      }
    #                           )

