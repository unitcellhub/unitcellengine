import numpy as np
from scipy import optimize
import random

def fedorovForm(Cin):
    """ Convert to Fredrov Voigt notation form 
    
    Convert from Voigt notation using tensorial strain to Fredrov form.

    Also known as Mandel or Kelvin notation.

    Reference
    ---------
    Nordmann, J., Aßmus, M., & Altenbach, H. (2018). Visualising elastic
    anisotropy: theoretical background and computational implementation.
    Continuum Mechanics and Thermodynamics, 30(4), 689–708.
    https://doi.org/10.1007/s00161-018-0635-9 
    """

    C = Cin.copy()
    C[:3, 3:] *= np.sqrt(2)/2
    C[3:, :3] *= np.sqrt(2)/2
    return C

def _d(phi, theta):
    """ Loading direction vector based on spherical angles
    
    Arguments
    ---------
    phi:   [0, pi]
    theta: [0, 2 pi]
    
    
    Theory
    ------
    d = [sin(φ)cos(θ),sin(φ)sin(θ),cos(φ)]

    Returns
    -------
    d: length 3 array corresponding to the specified loading direction
    """
    return np.array([np.sin(phi)*np.cos(theta), 
                      np.sin(phi)*np.sin(theta), 
                      np.cos(phi)])

def _n(phi, theta, psi):
    """ Normal vector to loading direction vector based on spherical angles
    
    Arguments
    ---------
    phi:   [0, pi]
    theta: [0, 2 pi]
    psi:   [0, 2 pi]

    Theory
    ------
    n = [sin(θ)cos(ψ)-cos(φ)cos(θ)cos(ψ),
         -cos(θ)sin(ψ)-cos(φ)sin(θ)cos(ψ)
         sin(φ)cos(ψ)]
    
    Returns
    -------
    n: length 3 array corresponding to the specified normal vector to loading direction
    """
    return np.array([np.sin(theta)*np.sin(psi)-np.cos(theta)*np.cos(phi)*np.cos(psi), 
                      -np.cos(theta)*np.sin(psi)-np.sin(theta)*np.cos(phi)*np.cos(psi), 
                      np.sin(phi)*np.cos(psi)])

def Ei(C, d):
    """ Calculate the elastic modulus in a specific direction 
    
    Arguments
    ---------
    C: 6x6 numpy array
        Voigt notation stiffness matrix for tensorial strains
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

    assert len(d) == 3, "Input direction vector must be length 3"
    d = (np.array(d)/np.linalg.norm(d)).reshape(3)

    # Convert to the Voigt notation used in the reference document
    S = np.linalg.inv(fedorovForm(C))

    # Calculate the dv vector
    d1, d2, d3 = d
    dv = np.array([[d1*d1, d2*d2, d3*d3, d1*d2, d2*d3, d3*d1]]).T
    dv[3:] *= np.sqrt(2)

    return (1/(dv.T @ S @ dv))[0, 0]

def Ki(C, d):
    """ Calculate the bulk modulus in a specific direction 
    
    Arguments
    ---------
    C: 6x6 numpy array
        Voigt notation stiffness matrix for tensorial strains
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

    assert len(d) == 3, "Input direction vector must be length 3"
    d = np.array(d)/np.linalg.norm(d)

    # Convert to the Voigt notation used in the reference document
    S = np.linalg.inv(fedorovForm(C))

    # Calculate the dv vector
    d1, d2, d3 = d
    dv = np.array([[d1*d1, d2*d2, d3*d3, d1*d2, d2*d3, d3*d1]]).T
    dv[3:] *= np.sqrt(2)

    # Create I vector
    Iv = np.array([[1]*3+[0]*3])

    return (1/(3*Iv @ S @ dv))[0, 0]

def Gij(C, *args):
    """ Calculate the bulk modulus in a specific orientation 
    
    Arguments
    ---------
    C: 6x6 numpy array
        Voigt notation stiffness matrix for tensorial strains
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

    if len(args) == 2:
        d, n = args
        assert len(d) == 3, "Input direction vector must be length 3"
        assert len(n) == 3, "Input normal vector must be length 3"
        assert np.isclose(np.inner(d, n), 0), "Normal vector must be " +\
                                            "perpendicular to the " +\
                                            "direction vector."
        # Make sure vectors are unit vectors
        d = np.array(d)/np.linalg.norm(d)
        n = np.array(n)/np.linalg.norm(n)
    elif len(args) == 3:
        phi, theta, psi = args
        assert 0 <= phi <= np.pi, "Phi must be between 0 and pi, not" +\
                                    f"{phi}."
        assert 0 <= theta <= 2*np.pi, "Theta must be between 0 and 2pi, "+\
                                        f"not {theta}."
        assert 0 <= psi <= 2*np.pi, "Psi must be between 0 and 2pi, not" +\
                                    f"{psi}."
        
        d = _d(phi, theta)
        n = _n(phi, theta, psi)
    else:
        raise ValueError("Incorrect input arguments. Must either "
                            "be in the form of d, n or phi, theta, psi.")

    # Convert to the Voigt notation used in the reference document
    S = np.linalg.inv(fedorovForm(C))

    # Calculate the dv vector
    d1, d2, d3 = d
    n1, n2, n3 = n
    mv = np.array([[d1*n1, d2*n2, d3*n3, 
                    d1*n2+d2*n1, d2*n3+d3*n2, d3*n1+d1*n3]]).T
    mv[:3] *= np.sqrt(2)

    return (1/(2*mv.T @ S @ mv))[0, 0]

def nuij(C, *args):
    """ Calculate the Poisson's ratio in a specific orientation 
    
    Arguments
    ---------
    C: 6x6 numpy array
        Voigt notation stiffness matrix for tensorial strains
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

    if len(args) == 2:
        d, n = args
        assert len(d) == 3, "Input direction vector must be length 3"
        assert len(n) == 3, "Input normal vector must be length 3"
        assert np.isclose(np.inner(d, n), 0), "Normal vector must be " +\
                                            "perpendicular to the " +\
                                            "direction vector."
        # Make sure vectors are unit vectors
        d = np.array(d)/np.linalg.norm(d)
        n = np.array(n)/np.linalg.norm(n)
    elif len(args) == 3:
        phi, theta, psi = args
        assert 0 <= phi <= np.pi, "Phi must be between 0 and pi, not" +\
                                    f"{phi}."
        assert 0 <= theta <= 2*np.pi, "Theta must be between 0 and 2pi, "+\
                                        f"not {theta}."
        assert 0 <= psi <= 2*np.pi, "Psi must be between 0 and 2pi, not" +\
                                    f"{psi}."
        
        d = _d(phi, theta)
        n = _n(phi, theta, psi)
    else:
        raise ValueError("Incorrect input arguments. Must either "
                            "be in the form of d, n or phi, theta, psi.")

    # Make sure vectors are unit vectors
    d = np.array(d)/np.linalg.norm(d)
    n = np.array(n)/np.linalg.norm(n)

    # Convert to the Voigt notation used in the reference document
    S = np.linalg.inv(fedorovForm(C))

    # Calculate the dv vector
    d1, d2, d3 = d
    n1, n2, n3 = n
    dv = np.array([[d1*d1, d2*d2, d3*d3, d1*d2, d2*d3, d3*d1]]).T
    dv[3:] *= np.sqrt(2)
    nv = np.array([[n1*n1, n2*n2, n3*n3, n1*n2, n2*n3, n3*n1]]).T
    nv[3:] *= np.sqrt(2)

    return (-Ei(C, d)*(dv.T @ S @ nv))[0, 0]

def _iext(ifun):
    """ Calculate the extreme values for (d) function 
    
    Arguments
    ---------
    ijfun: function handle
        Either the elastic modulus or bulk modulus function that
        takes in the arguments (phi, theta), which correspond to
        the loading direction.
    
    Returns
    -------
    Dictionary with keys "min" and "max", each of which contains a
    dictionary defining the "value", "d" direction vector, and "n"
    normal vector.
    """

    out = {}
    for sign, name in zip([1, -1], ['min', 'max']):
        fun = lambda x: sign*ifun(*x)
        sol = optimize.differential_evolution(fun, [(0, np.pi),
                                                    (0, 2*np.pi)])
        assert sol['success'], f"Searching for the {name} value " +\
                                "failed."
        phi, theta = sol['x']
        d = _d(phi, theta)
        out[name] = dict(value=sign*sol['fun'], d=d)
    
    return out

def _ijext(ijfun):
    """ Calculate the extreme values for (d, n) function 
    
    Arguments
    ---------
    ijfun: function handle
        Either the shear modulus or Poisson's ratio function that
        takes in the arguments (phi, theta, psi), which correspond to
        the loading direction and normal vector orientations
    
    Returns
    -------
    Dictionary with keys "min" and "max", each of which contains a
    dictionary defining the "value", "d" direction vector, and "n"
    normal vector.
    """

    out = {}
    for sign, name in zip([1, -1], ['min', 'max']):
        fun = lambda x: sign*ijfun(*x)
        sol = optimize.differential_evolution(fun, [(0, np.pi),
                                                    (0, 2*np.pi),
                                                    (0, 2*np.pi)])
        assert sol['success'], f"Searching for the {name} shear " +\
                                "modulus failed."
        phi, theta, psi = sol['x']
        d = _d(phi, theta)
        n = _n(phi, theta, psi)
        out[name] = dict(value=sign*sol['fun'], d=d, n=n)
    
    return out

#@jit(nopython=True, cache=True)
def _Eext(St):
    """ Calculate the extreme values for the elastic modulus

    Arguments
    ---------
    St: 3x3x3x3 numpy array
        4th order compliance tensor

    Returns
    -------
    (Emin, dmin), (Emax, dmax), where Exxx is the elastic
    modulus and dxxx is the corresponding loading direction.

    
    This is just-in-time compiled with Numba to vastly decrease
    execution time. This comes at the cost of missing a few of pythons
    built in features, but is worth it overall.
    """
    # Maximum number of iterations
    Nmax = 1000

    # Maximum number of repeat attempts for failed eigen solves
    Pmax = 100
    
    # Relative tolerance criteria
    rtol = 1e-5
    
    # Use the power method to find the maximum eigenvalue 
    counterMajor = 0
    while counterMajor < Pmax:
        # Generate a random initial guess at the eigen vector
        d0 = np.array([random.random(), random.random(), random.random()])
        d0 = d0/np.linalg.norm(d0)

        # Calculate the corresponding initial eigenvalue
        lam0 = np.einsum('ijkl,i,j,k,l', St, d0, d0, d0, d0)
        # lam0 = 0
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             for l in range(3):
        #                 lam0 += St[i, j, k, l]*d0[i]*d0[j]*d0[k]*d0[l]

        # Use the power method algorithm to find the maximum eigenvalue
        counter = 0
        while counter < Nmax:
            d = np.einsum('ijkl,j,k,l->i', St, d0, d0, d0)
            # d = np.zeros(3)
            # for j in range(3):
            #     for k in range(3):
            #         for l in range(3):
            #             d += St[:, j, k, l]*d0[j]*d0[k]*d0[l]
            
            # Convert d back to an unit vector and make sure it is in the 1st quadrant
            d = d/np.linalg.norm(d)
            
            # Calculate the corresponding eigenvalue
            lam = np.einsum('ijkl,i,j,k,l', St, d, d, d, d)
            # lam = 0
            # for i in range(3):
            #     for j in range(3):
            #         for k in range(3):
            #             for l in range(3):
            #                 lam += St[i, j, k, l]*d[i]*d[j]*d[k]*d[l]

            # Check for relative convergence
            if abs((lam-lam0)/lam0) < rtol:
                break
            
            # The eigenvalue should always be increasing. If it isn't,
            # mark as failed
            # if lam < lam0:
            #     counter = Nmax
            #     break

            # Store the current result for the next iteration
            lam0 = lam
            d0 = d
            counter += 1
        # Check to see if the last eigensolve converged. If so,
        # break out of the loop successfully
        if counter < Nmax:
            break
        counterMajor += 1
    assert counterMajor < Pmax, "Unable to find the maximum eigenvalue " +\
                                "using multiple initial conditions."
    # assert counter < Nmax, "Minimum elastic modulus calculation did " +\
    #                         "not converge"
    
    # The elastic modulus is the inverse of the quantity calculated
    # in the above eigenvalue problem. So, the result from this
    # max eigenvalue problem corresponds to the minimum elastic
    # modulus.
    Emin = 1/lam
    dmin = d

    # Find min eigenvalue using shift methodology
    

    # To solve for the minimum eigen vector using the power method,
    # shift the compliance tensor by the maximum eigenvalue.
    # All eigenvalues to this problem are negative, with the maximum
    # value corresponding to the minimum eigenvalue for the problem
    # of interest.
    I = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if i == k and j == l:
                        I[i, j, k, l] = 1
    counter = 0
    shifted = St - lam*I

    # Solver for the minimum eigenvector
    counterMajor = 0
    while counterMajor < Pmax:
        # Generate a random initial guess at the eigen vector
        d0 = np.array([random.random(), random.random(), random.random()])
        d0 = d0/np.linalg.norm(d0)

        # Calculate the corresponding initial eigenvalue
        lam0 = np.einsum('ijkl,i,j,k,l', St, d0, d0, d0, d0)
        # lam0 = 0
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             for l in range(3):
        #                 lam0 += St[i, j, k, l]*d0[i]*d0[j]*d0[k]*d0[l]

        # Use the power method algorithm to solve for the min eigen
        # value
        counter = 0
        while counter < Nmax:
            d = np.einsum('ijkl,j,k,l', shifted, d0, d0, d0)
            # d = np.zeros(3)
            # for j in range(3):
            #     for k in range(3):
            #         for l in range(3):
            #             d += shifted[:, j, k, l]*d0[j]*d0[k]*d0[l]
            
            # Convert d back to an unit vector and make sure it is in the 1st quadrant
            d = d/np.linalg.norm(d)

            # Calculate the corresponding eigenvalue
            lam = np.einsum('ijkl,i,j,k,l', St, d, d, d, d)
            # lam = 0
            # for i in range(3):
            #     for j in range(3):
            #         for k in range(3):
            #             for l in range(3):
            #                 lam += St[i, j, k, l]*d[i]*d[j]*d[k]*d[l]
            
            # Check for relative convergence
            if abs((lam-lam0)/lam0) < rtol:
                break

            # The eigenvalue should always be decreasing. If it isn't,
            # mark as failed
            # if lam > lam0:
            #     counter = Nmax
            #     break

            # Store the current result for the next iteration
            lam0 = lam
            d0 = d
            counter += 1

        # Check to see if the last eigensolve converged. If so,
        # break out of the loop successfully
        if counter < Nmax:
            break
        counterMajor += 1
    assert counterMajor < Pmax, "Unable to find the minimum eigenvalue " +\
                                "using multiple initial conditions."
    # assert counter < Nmax, "Maximum elastic modulus calculation did " +\
    #                         "not converge"
    
    # The elastic modulus is the inverse of the quantity calculated
    # in the above eigenvalue problem. So, the result from this
    # min eigenvalue problem corresponds to the max elastic
    # modulus.
    Emax = 1/lam
    dmax = d
    
    return (Emin, dmin), (Emax, dmax)

# @jit(nopython=True, cache=True)
def _Gext(St):
    """ Calculate the extreme values for the shear modulus

    Arguments
    ---------
    St: 3x3x3x3 numpy array
        4th order compliance tensor

    Returns
    -------
    (Gmin, dmin, nmain), (Gmax, dmax, nmax), where Gxxx is the shear
    modulus, dxxx is the corresponding loading direction, and nxxx is
    the corresponding normal direction.

    
    This is just-in-time compiled with Numba to vastly decrease
    execution time. This comes at the cost of missing a few of pythons
    built in features, but is worth it overall.
    """
    # Relative tolerance criteria
    rtol = 1e-5

    # Maximum number of eigen solve iterations
    Nmax = 2000

    # Maximum number of repeat attempts for failed eigen solves
    Pmax = 100

    # def mu(d, n):
    #     return np.einsum('ij,ijkl,k,l', 
    #                       np.outer(d, d)-np.outer(n, n), St, d, n)

    # def unit(x):
    #     return sum(x*x) - 1
    
    # def perp(x, y):
    #     return np.dot(x, y)

    # def dfuns(d, n):
    #     return np.array([mu(d, n), unit(d), perp(d, n)])
    
    # def nfuns(n, d):
    #     return np.array([mu(d, n), unit(n), perp(d, n)])
    
    # def ndsol(n, d, nd):
    #     return (n+d) - nd

    # def ndfuns(x, nd):
    #     n = x[:3]
    #     d = x[3:]
    #     return np.hstack([ndsol(n, d, nd),
    #                      np.array([unit(d), unit(n), perp(n, d)])])

    # Solve for the max eigenvalue
    counterMajor = 0
    while counterMajor < Pmax:
        # Create a random unit direction vector as an initial
        # solution guess
        d0 = np.array([random.random(), random.random(), random.random()])
        d0 = d0/np.linalg.norm(d0)

        # Create a random unit normal vector to the created
        # direction vector
        n0 = np.cross(d0, 
                    np.array([random.random(), random.random(), random.random()]))
        n0 = n0/np.linalg.norm(n0)

        # Solve for the maximum eigen direction
        counter = 0
        # mu0 = 0.5*np.einsum('ijkl,i,j,k,l', St, nd0, nd0, d0, n0)
        mu0 = np.einsum('ijkl,i,j,k,l', St, n0+d0, n0+d0, d0, n0)
        # lam0 = np.einsum('ijkl,i,j,k,l', 4*St, d0, n0, d0, n0)
        # lam0 = 0
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             for l in range(3):
        #                 lam0 += 4*St[i, j, k, l]*d0[i]*n0[j]*d0[k]*n0[l]
        while counter < Nmax:
            # Update d, ensuring unicity and perpendicularity to n
            d = np.einsum('ijkl,j,k,l->i', St, n0, d0, n0)
            d = d/np.linalg.norm(d)
            n0 = np.cross(np.cross(d, n0), d)
            n0 = n0/np.linalg.norm(n0)
            

            # Update n, ensuring unicity and perpendicularity to d
            n = np.einsum('ijkl,i,k,l->j', St, d, d, n0)
            n = n/np.linalg.norm(n)
            d = np.cross(np.cross(n, d), n)
            d = d/np.linalg.norm(d)
            

            # Check for relative convergence. Note, due to the
            # ambiguity of the vectors, we check for vectors in the
            # opposite direction (which are equally valid)
            # Update n now
            # lam = np.einsum('ijkl,i,j,k,l', 4*St, d, n, d, n)
            # mu = 0.5*np.einsum('ijkl,i,j,k,l', St, nd, nd, d, n)
            mu = np.einsum('ijkl,i,j,k,l', St, n+d, n+d, d, n)
            # lam = 0
            # for i in range(3):
            #     for j in range(3):
            #         for k in range(3):
            #             for l in range(3):
            #                 lam += 4*St[i, j, k, l]*d[i]*n[j]*d[k]*n[l]
            
            if abs((mu-mu0)/mu0) < rtol:
                break

            # The eigenvalue should always be increasing. If it isn't,
            # mark as failed
            # if lam < lam0:
            #     counter = Nmax
            #     break

            # Store iteration results 
            mu0 = mu
            # nd0 = nd
            # lam0 = lam
            d0 = d
            n0 = n
            counter += 1
        
        # Check to see if the last eigensolve converged. If so,
        # break out of the loop successfully
        if counter < Nmax:
            break
        counterMajor += 1
    assert counterMajor < Pmax, "Unable to find the maximum eigenvalue " +\
                                "using multiple initial conditions."
    
    # Calculate the minimum shear modulus, noting that it is the
    # inverse of the eigen value we just solved for
    Gmin = 1/np.einsum('ijkl,i,j,k,l', 4*St, d, n, d, n)
    dmin = d
    nmin = n

    # Find min eigenvalue using shift methodology
    counterMajor = 0
    while counterMajor < Pmax:
        # Create a random unit direction vector as an initial
        # solution guess
        d0 = np.array([random.random(), random.random(), random.random()])
        d0 = d0/np.linalg.norm(d0)
        # Create a random unit normal vector to the created
        # direction vector
        n0 = np.cross(d0, 
                    np.array([random.random(), random.random(), random.random()]))
        n0 = n0/np.linalg.norm(n0)
        
        # Solve the minimum eigen value problem using the shifted
        # power method
        counter = 0
        I = np.zeros((3, 3, 3, 3)) 
        # 1 if i == k and j == l, otherwise 0
        I[0, 0, 0, 0] = I[0, 1, 0, 1] = I[0, 2, 0, 2] = 1
        I[1, 0, 1, 0] = I[1, 1, 1, 1] = I[1, 2, 1, 2] = 1
        I[2, 0, 2, 0] = I[2, 1, 2, 1] = I[2, 2, 2, 2] = 1
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             for l in range(3):
        #                 if i == k and j == l:
        #                     assert I[i, j, k, l] == 1
        #                     # I[i, j, k, l] = 1 
        counter = 0
        # Shift the tensor by the maximum eigen value to convert thep
        # the problem to a solution for the minimum eigen value
        shifted = St - mu*I
        mu0 = np.einsum('ijkl,i,j,k,l', shifted, n0+d0, n0+d0, d0, n0)
        # lam0 = np.einsum('ijkl,i,j,k,l', 4*St, d0, n0, d0, n0)
        # lam0 = 0
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             for l in range(3):
        #                 lam0 += 4*St[i, j, k, l]*d0[i]*n0[j]*d0[k]*n0[l]
        while counter < Nmax:
            # Update d, ensuring unicity and perpendicularity to n
            d = np.einsum('ijkl,j,k,l->i', shifted, n0, d0, n0)
            d = d/np.linalg.norm(d)
            n0 = np.cross(np.cross(d, n0), d)
            n0 = n0/np.linalg.norm(n0)
            
            # Update n, ensuring unicity and perpendicularity to d
            n = np.einsum('ijkl,i,k,l->j', shifted, d, d, n0)
            n = n/np.linalg.norm(n)
            d = np.cross(np.cross(n, d), n)
            d = d/np.linalg.norm(d)

            # Check for relative convergence. Note, due to the
            # ambiguity of the vectors, we check for vectors in the
            # opposite direction (which are equally valid)
            # Update n now
            # lam = np.einsum('ijkl,i,j,k,l', 4*St, d, n, d, n)
            # mu = 0.5*np.einsum('ijkl,i,j,k,l', St, nd, nd, d, n)
            mu = np.einsum('ijkl,i,j,k,l', shifted, n+d, n+d, d, n)
            # lam = 0
            # for i in range(3):
            #     for j in range(3):
            #         for k in range(3):
            #             for l in range(3):
            #                 lam += 4*St[i, j, k, l]*d[i]*n[j]*d[k]*n[l]
            
            if abs((mu-mu0)/mu0) < rtol:
                break
            
            # The eigenvalue should always be decreasing. If it isn't,
            # mark as failed
            # if lam > lam0:
            #     counter = Nmax
            #     break

            # Update the reference fectors
            # lam0 = lam
            mu0 = mu
            d0 = d
            n0 = n
            counter += 1

        # Check to see if the last eigensolve converged. If so,
        # break out of the loop successfully
        if counter < Nmax:
            break
        counterMajor += 1
    assert counterMajor < Pmax, "Unable to find the minimum eigenvalue " +\
                                 "using multiple initial conditions."

    # Calculate the maximum shear modulus, noting that it
    # corresponds to the inverse of the eigen value solution
    Gmax = 1/np.einsum('ijkl,i,j,k,l', 4*St, d, n, d, n)
    dmax = d
    nmax = n

    return (Gmin, dmin, nmin), (Gmax, dmax, nmax)
