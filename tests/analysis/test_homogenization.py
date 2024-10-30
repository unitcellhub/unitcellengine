import unittest
import unitcellengine.analysis.homogenization as homog
from pathlib import Path
import numpy as np
import math

class TestElasticHomogenization(object):
    """ Test elastic homogenization method """

    module = None
    format = None

    def testFullyDense(self):
        """ Compare fully dense homogenization with elasticity """

        # Define mesh file
        basename = Path(__file__).parent
        mesh =  basename / Path('resources/fullyDense').with_suffix(self.format)

        # Create homogenization instance, run simulations, and process
        h = self.module(mesh, E=1, nu=0.3)
        h.clear() # Make sure to clear old results files
        h.appliedStrain = 0.0005
        self.assertTrue(h.run(reuse=False, blocking=True))

        # Make sure all the runs completed correctly
        # self.assertDictEqual(h.check(), {i: True for i in range(1, 7)})

        # Post process the results
        h.process(check=True, reuse=False, rtol=1e-2)
        CH = h.CH

        # Compare against expected constitutive behavior for a fully
        # dense elastic material.
        E = h.E
        nu = h.nu
        lam = E*nu/((1+nu)*(1-2*nu))
        mu = 0.5*E/(1+nu)
        exp = np.zeros((6, 6))
        exp[0, 0] = exp[1, 1] = exp[2, 2] = lam+2*mu
        exp[3, 3] = exp[4, 4] = exp[5, 5] = mu*2
        exp[0, 1] = exp[0, 2] = exp[1, 2] = lam
        exp[1, 0] = exp[2, 0] = exp[2, 1] = lam
        
        self.assertTrue(np.allclose(CH, exp, rtol=1e-2, atol=E*1e-4))
        self.assertTrue(math.isclose(h.anisotropyIndex, 0, abs_tol=1e-4))

        # Check that mean strains corresponds to uniform strain
        # throughout
        for averageStrain in [np.array([0.1, 0, 0, 0, 0, 0]), 
                             np.array([0, 0, 0, 0, 0.1, 0])]:
            strains = h.localStrain(averageStrain)
            expStrain = np.tile(averageStrain.reshape(-1, 1), 
                                (1, strains.shape[1]))
            self.assertTrue(np.allclose(strains, expStrain))

    def testOctet(self):
        """ Compare octet with values from the literature 
        
        Ref: https://asa.scitation.org/doi/10.1121/1.5091690

        Vxx = sqrt(C11/rho)
        Vxy = Vxz = sqrt(C66/rho)
        2 rho V45^2 = (C11 + C66) ± (C12 + C66)

        where rho is the density of the solid material.

        Note: all tabulated values are for E = 1 GPa, nu = 0.35 and rho
        = 1097 kg/m^3

        Reference geometry: relative density = 0.531465

        To create the geomtry: SDFGeometry("Octet", 1, 1, 1, 0.2525, form="graph")

        Excpected values:
        C11 = 0.31742 GPa
        C12 = 0.14456 GPa
        C44 = 0.12014 GPa (noting that this is half the value that is
        calculated due to the difference between engineering and
        tensorial shear strain)


        Note, these expected values were manually pulled from a graph in
        a paper, so they aren't super accurate. The corresponding
        comparisons are therefore looking for coarse similarity.
        """

        # Define mesh file
        basename = Path(__file__).parent
        mesh =  basename / Path('resources/octetPatil').with_suffix(self.format)

        # Create homogenization instance, run simulations, and process
        
        h = self.module(mesh, E=1, nu=0.35)
        h.clear() # Make sure to clear old results files
        self.assertTrue(h.run(reuse=False, blocking=True))

        # Make sure all the runs completed correctly
        # self.assertDictEqual(h.check(), {i: True for i in range(1, 7)})

        # Post process the results
        h.process(check=True, reuse=False, rtol=1e-2)
        CH = h.CH

        # Compare against expected constitutive behavior for a fully
        # dense elastic material.
        exp = np.zeros((6, 6))
        exp[0, 0] = exp[1, 1] = exp[2, 2] = 0.31742
        exp[3, 3] = exp[4, 4] = exp[5, 5] = 0.12014*2
        exp[0, 1] = exp[0, 2] = exp[1, 2] = 0.14456
        exp[1, 0] = exp[2, 0] = exp[2, 1] = 0.14456
        
        self.assertTrue(np.allclose(CH, exp, rtol=1e-3, atol=5e-2)) 

    def testHexHoneycomb(self):
        """ Validate honeycomb stiffnesses and stresses 
        
        Validate against Gibson, et al "The Mechanics of two-dimensional
        cellular materials". Note that the analytical calculations are
        based on slightly different theory and in the context of beams
        rather than 3D elasticity; so, the validation hear is relatively
        rough, but we still expect similar order of magnitude results.

        Relative density
        rho/rho_s = [(2+h/l)t/l]/[2 cosθ (h/l+sinθ)]

        Stiffnesses
        E1/Es = 12 I cosθ/[(h+l sinθ)b l^2 sin^2θ]
        E2/Es = 12 I (h + l sinθ)/[b l^4 cos^3θ]
        E3/Es = ρ_s/ρ

        where I = t^3 b/12, l is the diagonal length, h is the vertical
        section length, t is the ligament thickness, b is the ligament
        depth, and θ is the diagonal ligament angle relative to the
        horizontal direction.

        Geometry
        --------
        Note, the directional correspondence is as follows:
        Theory -> code
        - 1 -> x
        - 2 -> y
        - 3 -> z

           ^ 2 direction
           |       
        \ /\ /
         |  |   -> 1 direction
        / \/ \


        Stresses in 1 direction
        ----------------------
        M = P l sinθ/2
        P = σ1(h + l sinθ)b
        σ_bending = MI/[t/2] + P l I sinθ/t
        σ_axial = P cosθ/(tb)

        Stresses in 2 direction
        ----------------------
        M = W l cosθ/2
        W = σ2 lb cosθ
        σ_bending = MI/[t/2] + W l I cosθ/t
        σ_axial = P sinθ/(tb)

        Stresses in direction 3
        ----------------------=
        σ = ρ_s/ρ σ3

        """

        # Define mesh file
        basename = Path(__file__).parent
        mesh =  basename / Path('resources/hexHoneycomb_5x5x1').with_suffix(self.format)

        # Create homogenization instance, run simulations, and process
        E = 1
        
        job = self.module(mesh, E=E, nu=0.35)
        job.run(reuse=False, blocking=True, nprocessors=8)
        

        # Post process the results and pull out relevant results
        job.process(check=True, reuse=False, rtol=1e-2)
        vm11 = job.result['amplification']['vm11']
        vm22 = job.result['amplification']['vm22']
        vm33 = job.result['amplification']['vm33']
        
        E1 = job.Ei([1, 0, 0])
        E2 = job.Ei([0, 1, 0])
        E3 = job.Ei([0, 0, 1])
        nu12 = job.nuij([1, 0, 0], [0, 1, 0])

        # Analytical values based on the mesh geometry
        # L = 2 l cosθ
        # W = 2 h + 2 l sinθ
        t = 3
        L = 5*10
        H = 1*10
        rel = 0.161085
        b = H
        theta = 0.280781
        I = t**3*b/12.
        ell = L/2./np.cos(theta)
        h = 17.79

        s = np.sin(theta)
        c = np.cos(theta)

        # Loading in the X direction
        # Expected peak stress is Mc/I + P cosθ/(bt)
        sig1 = 2.   # Note, that paper definition is half the value of ours
        P = sig1*(h+ell*s)*b
        M = 0.5*P*ell*s
        vm11_expected = M*t/2/I + P*c/(b*t)
        # print(vm11, vm11_expected)

        # Loading in the Y direction
        # Expected peak stress is Mc/I + P sinθ/(bt)
        sig2 = 2
        P = sig2*ell*b*c
        M = 0.5*P*ell*c
        vm22_expected = M*t/2/I + P*s/(b*t)
        # print(vm22, vm22_expected)

        # Loading in the Z direction
        # Expected to just be the ratio of the exposed surface area
        vm33_expected = 1/rel

        # Expected elastic properties
        E1_expected = 12*I*c*E/((h+ell*s)*b*ell**2*s**2)
        E2_expected = 12*I*E*(h + ell*s)/(b*ell**4*c)
        E3_expected = rel
        nu12_expected = ell*c**2/(h+ell*s)/s

        # Compare and assert
        self.assertTrue(math.isclose(E1, E1_expected, rel_tol=0.2))
        self.assertTrue(math.isclose(E2, E2_expected, rel_tol=0.4))
        self.assertTrue(math.isclose(E3, E3_expected, rel_tol=0.1))
        self.assertTrue(math.isclose(nu12, nu12_expected, rel_tol=0.25))
        self.assertTrue(math.isclose(vm11, vm11_expected, rel_tol=0.25))
        self.assertTrue(math.isclose(vm22, vm22_expected, rel_tol=0.50))
        self.assertTrue(math.isclose(vm33, vm33_expected, rel_tol=0.1))

    def testAnisotropyIndex(self):
        """ Validate anisotropy index calculation """

        # Define mesh file
        basename = Path(__file__).parent
        mesh =  basename / Path('resources/fullyDense').with_suffix(self.format)

        # Create homogenization instance
        
        h = self.module(mesh, E=1, nu=0.3)
        h.clear() # Make sure to clear old results files
        
        # Overide homogenization matrix with local elasticity matrix
        h.CH = h.C

        # Calculate anisotropy index
        AU = h.anisotropyIndex

        # Verify that it is 0 for this case (which is the value of the
        # anisotropy index for an isotropic material)
        self.assertTrue(math.isclose(AU, 0, abs_tol=1e-6))

    def testEngineeringConstants(self):
        """ Test the engineering constants (E, K, G, nu) calculations """

        # Define an arbitrary mesh file (which isn't actually used)
        basename = Path(__file__).parent
        mesh =  basename / Path('resources/fullyDense').with_suffix(self.format)
        
        h = self.module(mesh, E=1, nu=0.3)
        h.clear() # Make sure to clear old results files

        # Create an isotropic material and check against
        # Compare against expected constitutive behavior for a fully
        # dense elastic material.
        E = h.E
        nu = h.nu
        G = E/(2*(1+nu))
        K = E/(3*(1-2*nu))
        lam = E*nu/((1+nu)*(1-2*nu))
        mu = 0.5*E/(1+nu)
        exp = np.zeros((6, 6))
        exp[0, 0] = exp[1, 1] = exp[2, 2] = lam+2*mu
        exp[3, 3] = exp[4, 4] = exp[5, 5] = mu*2
        exp[0, 1] = exp[0, 2] = exp[1, 2] = lam
        exp[1, 0] = exp[2, 0] = exp[2, 1] = lam
        h.CH = exp

        # Run through a range of orientations and compare against
        # expected
        PHI, THETA = np.meshgrid(np.linspace(0, np.pi, 4), 
                                 np.linspace(0, 2*np.pi, 4))
        c = np.cos
        s = np.sin
        for phi, theta in zip(PHI.flatten(), THETA.flatten()):
            d = [s(phi)*c(theta), 
                 s(phi)*s(theta),
                 c(phi)]
            # Calculate single direction constants
            Ec = h.Ei(d)
            Kc = h.Ki(d)
            self.assertTrue(math.isclose(Ec, E, abs_tol=1e-6))
            self.assertTrue(math.isclose(Kc, K, abs_tol=1e-6))

            # Calculate multi direction constants
            for psi in np.linspace(0, np.pi*2, 4):
                n = [s(theta)*s(psi) - c(phi)*c(theta)*c(psi), 
                    -c(theta)*s(psi) - c(phi)*s(theta)*c(psi),
                     s(phi)*c(psi)]

                # Skip colinear vectors
                if np.isclose(np.linalg.norm(np.cross(n, d)), 0):
                    continue

                Gc = h.Gij(d, n)
                nuc = h.nuij(d, n)
                self.assertTrue(math.isclose(Gc, G, abs_tol=1e-6))
                self.assertTrue(math.isclose(nuc, nu, abs_tol=1e-6))
    
        # Compare against results from Nordmann, J., Aßmus, M., &
        # Altenbach, H. (2018). Visualising elastic anisotropy:
        # theoretical background and computational implementation.
        # Continuum Mechanics and Thermodynamics, 30(4), 689–708.
        # https://doi.org/10.1007/s00161-018-0635-9.
        # Note that all of the comparisons are relatively coarse as the
        # validation numbers are pulled from a colorbar.
        exp = np.zeros((6, 6))
        exp[0, 0] = exp[1, 1] = exp[2, 2] = 165.7
        exp[3, 3] = exp[4, 4] = exp[5, 5] = 79.6*2
        exp[0, 1] = exp[0, 2] = exp[1, 2] = 63.9
        exp[1, 0] = exp[2, 0] = exp[2, 1] = 63.9
        h.CH = exp 

        # From Figure 2
        Eext = h.Eext()
        self.assertTrue(math.isclose(h.Ei([1, 0, 0]), 135, abs_tol=5))
        self.assertTrue(math.isclose(h.Ei([1, 1, 1]), 185, abs_tol=5))
        self.assertTrue(math.isclose(Eext['min']['value'], 135, abs_tol=5))
        self.assertTrue(math.isclose(Eext['max']['value'], 185, abs_tol=5))
        
        # From Figure 3
        Kext = h.Kext()
        self.assertTrue(math.isclose(h.Ki([1, 0, 0]), 97.83, abs_tol=1e-2))
        self.assertTrue(math.isclose(h.Ki([0, 1, 0]), 97.83, abs_tol=1e-2))
        self.assertTrue(math.isclose(h.Ki([1, 1, 0]), 97.83, abs_tol=1e-2))
        self.assertTrue(math.isclose(h.Ki([1, 1, 1]), 97.83, abs_tol=1e-2))
        self.assertTrue(math.isclose(Kext['min']['value'], 97.83, abs_tol=5))
        self.assertTrue(math.isclose(Kext['max']['value'], 97.83, abs_tol=5))
        
        # From Figure 4
        # Note: T rotates vector v about axis r by theta
        Gext = h.Gext()
        T = lambda v, r, theta: (1-c(theta))*(np.inner(v, r))*r +\
                                  c(theta)*v + \
                                  s(theta)*np.cross(r, v)
        self.assertTrue(math.isclose(h.Gij([1, 0, 0], [0, 0, 1]), 77.5, 
                                     abs_tol=2.5))
        self.assertTrue(math.isclose(h.Gij([1, 0, 0], [0, 1, 0]), 77.5, 
                                     abs_tol=2.5))
        d = np.array([1, 1, 0])
        n = np.array([0, 0, 1])
        minG = min([h.Gij(d, T(n, d, theta)) for theta in np.linspace(0, 2*np.pi, 100)])
        self.assertTrue(math.isclose(minG, 50, abs_tol=2.5))
        self.assertTrue(math.isclose(h.Gij([1, 1, 1], [0, 1, -1]), 60, 
                                     abs_tol=2.5))
        self.assertTrue(math.isclose(h.Gij([1, 1, 1], [0, -1, 1]), 60, 
                                     abs_tol=2.5))
        self.assertTrue(math.isclose(Gext['min']['value'], 52, abs_tol=5))
        self.assertTrue(math.isclose(Gext['max']['value'], 77.5, abs_tol=5))

        # From Figure 5
        def nuext(d, n0, fun):
            """ Find max nu for given direction """
            d = np.array(d)
            n0 = np.array(n0)
            out = fun([h.nuij(d, T(n0, d, theta)) for 
                                theta in np.linspace(0, 2*np.pi, 100)])
            return out
        self.assertTrue(math.isclose(nuext([0, 1, 1], [1, 0, 0], max), 0.36, 
                                     abs_tol=0.025))
        self.assertTrue(math.isclose(nuext([0, 1, 1], [1, 0, 0], min), 0.08, 
                                     abs_tol=0.025))
        self.assertTrue(math.isclose(nuext([0, 1, 0], [1, 0, 0], min), 0.27, 
                                     abs_tol=0.025))
        self.assertTrue(math.isclose(nuext([0, 0, 1], [1, 0, 0], min), 0.27, 
                                     abs_tol=0.025))
        self.assertTrue(math.isclose(nuext([1, 1, 1], [1, -1, 0], max), 0.17, 
                                     abs_tol=0.025))
        nuext = h.nuext()
        self.assertTrue(math.isclose(nuext['min']['value'], 0.08, abs_tol=0.025))
        self.assertTrue(math.isclose(nuext['max']['value'], 0.36, abs_tol=0.025))

class TestConductanceHomogenization(object):
    """ Test elastic homogenization method """

    module = None
    format = None

    def testFullyDense(self):
        """ Compare fully dense homogenization with conductance """

        # Define mesh file
        basename = Path(__file__).parent
        mesh =  basename / Path('resources/fullyDense').with_suffix(self.format)

        # Create homogenization instance, run simulations, and process
        
        h = self.module(mesh, k=1)
        h.clear() # Make sure to clear old results files
        self.assertTrue(h.run(reuse=False, blocking=True))

        # Make sure all the runs completed correctly
        # self.assertDictEqual(h.check(), {i: True for i in range(1, 7)})

        # Post process the results
        h.process(check=True, reuse=False, rtol=1e-2)
        CH = h.CH

        # Compare against expected constitutive behavior for a fully
        # dense elastic material.
        exp = np.eye(3)*h.k
        
        self.assertTrue(np.allclose(CH, exp, rtol=1e-2, atol=h.k*1e-4))

        # # Check that mean strains corresponds to uniform strain
        # # throughout
        # for averageStrain in [np.array([0.1, 0, 0, 0, 0, 0]), 
        #                      np.array([0, 0, 0, 0, 0.1, 0])]:
        #     strains = h.localStrain(averageStrain)
        #     expStrain = np.tile(averageStrain.reshape(-1, 1), 
        #                         (1, strains.shape[1]))
        #     self.assertTrue(np.allclose(strains, expStrain))

    def testTPMS(self):
        """ Compare gyroid with values from the literature 
        
        Ref: http://dx.doi.org/10.1016/j.commatsci.2014.12.039

        Effective conductances pulled from Figure 9.

        Gyroid: Thickness of 0.08
        Schwarz: Thickness of 0.1

        Note, these expected values were manually pulled from a graph in
        a paper, so they aren't super accurate. The corresponding
        comparisons are therefore looking for coarse similarity.
        """

        # Define mesh file
        basename = Path(__file__).parent
        
        exps = {"halesGyroid_0_24": 0.168,
                "halesSchwarz_0_23": 0.159}
        for resource, exp in exps.items():
            mesh =  basename / Path(f'resources/{resource}').with_suffix(self.format)

            # Create homogenization instance, run simulations, and process
            h = self.module(mesh, k=1)
            h.clear() # Make sure to clear old results files
            self.assertTrue(h.run(reuse=False, blocking=True))
            
            # Post process the results
            h.process(check=True, reuse=False, rtol=1e-2)
            CH = h.CH

            # Compare against expected constitutive behavior from 
            # literature
            self.assertTrue(np.isclose(CH[0, 0], CH[1, 1], rtol=1e-3, atol=5e-2)) 
            self.assertTrue(np.isclose(CH[0, 1], 0, rtol=1e-3, atol=5e-2)) 
            self.assertTrue(np.isclose(CH[1, 0], 0, rtol=1e-3, atol=5e-2)) 
            self.assertTrue(np.isclose(CH[0, 0], exp, rtol=1e-1, atol=5e-2))

#     def testEngineeringConstants(self):
#         """ Test the engineering constants (E, K, G, nu) calculations """

#         # Define an arbitrary mesh file (which isn't actually used)
#         basename = Path(__file__).parent
#         mesh =  basename / Path('resources/fullyDense').with_suffix(self.format)
#         h = self.module(mesh, E=1, nu=0.3)

#         # Create an isotropic material and check against
#         # Compare against expected constitutive behavior for a fully
#         # dense elastic material.
#         E = h.E
#         nu = h.nu
#         G = E/(2*(1+nu))
#         K = E/(3*(1-2*nu))
#         lam = E*nu/((1+nu)*(1-2*nu))
#         mu = 0.5*E/(1+nu)
#         exp = np.zeros((6, 6))
#         exp[0, 0] = exp[1, 1] = exp[2, 2] = lam+2*mu
#         exp[3, 3] = exp[4, 4] = exp[5, 5] = mu*2
#         exp[0, 1] = exp[0, 2] = exp[1, 2] = lam
#         exp[1, 0] = exp[2, 0] = exp[2, 1] = lam
#         h.CH = exp

#         # Run through a range of orientations and compare against
#         # expected
#         PHI, THETA = np.meshgrid(np.linspace(0, np.pi, 4), 
#                                  np.linspace(0, 2*np.pi, 4))
#         c = np.cos
#         s = np.sin
#         for phi, theta in zip(PHI.flatten(), THETA.flatten()):
#             d = [s(phi)*c(theta), 
#                  s(phi)*s(theta),
#                  c(phi)]
#             # Calculate single direction constants
#             Ec = h.Ei(d)
#             Kc = h.Ki(d)
#             self.assertTrue(math.isclose(Ec, E, abs_tol=1e-6))
#             self.assertTrue(math.isclose(Kc, K, abs_tol=1e-6))

#             # Calculate multi direction constants
#             for psi in np.linspace(0, np.pi*2, 4):
#                 n = [s(theta)*s(psi) - c(phi)*c(theta)*c(psi), 
#                     -c(theta)*s(psi) - c(phi)*s(theta)*c(psi),
#                      s(phi)*c(psi)]

#                 # Skip colinear vectors
#                 if np.isclose(np.linalg.norm(np.cross(n, d)), 0):
#                     continue

#                 Gc = h.Gij(d, n)
#                 nuc = h.nuij(d, n)
#                 self.assertTrue(math.isclose(Gc, G, abs_tol=1e-6))
#                 self.assertTrue(math.isclose(nuc, nu, abs_tol=1e-6))
    
#         # Compare against results from Nordmann, J., Aßmus, M., &
#         # Altenbach, H. (2018). Visualising elastic anisotropy:
#         # theoretical background and computational implementation.
#         # Continuum Mechanics and Thermodynamics, 30(4), 689–708.
#         # https://doi.org/10.1007/s00161-018-0635-9.
#         # Note that all of the comparisons are relatively coarse as the
#         # validation numbers are pulled from a colorbar.
#         exp = np.zeros((6, 6))
#         exp[0, 0] = exp[1, 1] = exp[2, 2] = 165.7
#         exp[3, 3] = exp[4, 4] = exp[5, 5] = 79.6*2
#         exp[0, 1] = exp[0, 2] = exp[1, 2] = 63.9
#         exp[1, 0] = exp[2, 0] = exp[2, 1] = 63.9
#         h.CH = exp 

#         # From Figure 2
#         Eext = h.Eext()
#         self.assertTrue(math.isclose(h.Ei([1, 0, 0]), 135, abs_tol=5))
#         self.assertTrue(math.isclose(h.Ei([1, 1, 1]), 185, abs_tol=5))
#         self.assertTrue(math.isclose(Eext['min']['value'], 135, abs_tol=5))
#         self.assertTrue(math.isclose(Eext['max']['value'], 185, abs_tol=5))
        
#         # From Figure 3
#         Kext = h.Kext()
#         self.assertTrue(math.isclose(h.Ki([1, 0, 0]), 97.83, abs_tol=1e-2))
#         self.assertTrue(math.isclose(h.Ki([0, 1, 0]), 97.83, abs_tol=1e-2))
#         self.assertTrue(math.isclose(h.Ki([1, 1, 0]), 97.83, abs_tol=1e-2))
#         self.assertTrue(math.isclose(h.Ki([1, 1, 1]), 97.83, abs_tol=1e-2))
#         self.assertTrue(math.isclose(Kext['min']['value'], 97.83, abs_tol=5))
#         self.assertTrue(math.isclose(Kext['max']['value'], 97.83, abs_tol=5))
        
#         # From Figure 4
#         # Note: T rotates vector v about axis r by theta
#         Gext = h.Gext()
#         T = lambda v, r, theta: (1-c(theta))*(np.inner(v, r))*r +\
#                                   c(theta)*v + \
#                                   s(theta)*np.cross(r, v)
#         self.assertTrue(math.isclose(h.Gij([1, 0, 0], [0, 0, 1]), 77.5, 
#                                      abs_tol=2.5))
#         self.assertTrue(math.isclose(h.Gij([1, 0, 0], [0, 1, 0]), 77.5, 
#                                      abs_tol=2.5))
#         d = np.array([1, 1, 0])
#         n = np.array([0, 0, 1])
#         minG = min([h.Gij(d, T(n, d, theta)) for theta in np.linspace(0, 2*np.pi, 100)])
#         self.assertTrue(math.isclose(minG, 50, abs_tol=2.5))
#         self.assertTrue(math.isclose(h.Gij([1, 1, 1], [0, 1, -1]), 60, 
#                                      abs_tol=2.5))
#         self.assertTrue(math.isclose(h.Gij([1, 1, 1], [0, -1, 1]), 60, 
#                                      abs_tol=2.5))
#         self.assertTrue(math.isclose(Gext['min']['value'], 52, abs_tol=5))
#         self.assertTrue(math.isclose(Gext['max']['value'], 77.5, abs_tol=5))

#         # From Figure 5
#         def nuext(d, n0, fun):
#             """ Find max nu for given direction """
#             d = np.array(d)
#             n0 = np.array(n0)
#             out = fun([h.nuij(d, T(n0, d, theta)) for 
#                                 theta in np.linspace(0, 2*np.pi, 100)])
#             return out
#         self.assertTrue(math.isclose(nuext([0, 1, 1], [1, 0, 0], max), 0.36, 
#                                      abs_tol=0.025))
#         self.assertTrue(math.isclose(nuext([0, 1, 1], [1, 0, 0], min), 0.08, 
#                                      abs_tol=0.025))
#         self.assertTrue(math.isclose(nuext([0, 1, 0], [1, 0, 0], min), 0.27, 
#                                      abs_tol=0.025))
#         self.assertTrue(math.isclose(nuext([0, 0, 1], [1, 0, 0], min), 0.27, 
#                                      abs_tol=0.025))
#         self.assertTrue(math.isclose(nuext([1, 1, 1], [1, -1, 0], max), 0.17, 
#                                      abs_tol=0.025))
#         nuext = h.nuext()
#         self.assertTrue(math.isclose(nuext['min']['value'], 0.08, abs_tol=0.025))
#         self.assertTrue(math.isclose(nuext['max']['value'], 0.36, abs_tol=0.025))


# class TestAdagioElasticHomogenization(TestElasticHomogenization,
#                                       unittest.TestCase):
#     module = homog.AdagioElasticHomogenization
#     format = ".e"          

class TestInternalElasticHomogenization(TestElasticHomogenization, 
                                        unittest.TestCase):
    module = homog.InternalElasticHomogenization
    format = ".npz"

class TestInternalConductanceHomogenization(TestConductanceHomogenization, 
                                            unittest.TestCase):
    module = homog.InternalConductanceHomogenization
    format = ".npz"

if __name__ == "__main__":
    unittest.main()
    # TestInternalElasticHomogenization()
