# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:11:38 2018

@author: watkinrt
"""

from setuptools import setup

setup(
    name="unitcell",
    version="0.0.1",
    description="Unitcell generation and evaluation",
    url="https://github.com/unitcellhub/unitcellengine",
    author="Ryan Watkins",
    author_email="ryan.t.watkins@jpl.nasa.gov",
    packages=[
        "unitcell",
        "unitcell.geometry",
        "unitcell.geometry.definitions",
        "unitcell.mesh",
        "unitcell.analysis",
    ],
    package_data={
        "unitcell.geometry.definitions": ["**/*.json"],
    },
    zip_safe=False,
    python_requires=">=3, !=3.11.*",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "trimesh",
        "pandas",
        "plotly",
        "meshio==4.3.11",
        "gmsh",
        "scikit-image",
        "numexpr",
        "blosc",
        "tables",
        "tqdm",
        "vtk",
        "pyvista",
        "numba",
        "mkl-service",
        "pypardiso",
        "pyamg",
        "sdf@git+https://github.com/fogleman/sdf@main#egg=sdf",
        "dill",
        "scikit-learn",
        "pyvista",
    ],
)
