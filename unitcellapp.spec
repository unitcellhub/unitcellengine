# -*- mode: python ; coding: utf-8 -*-

# Notes:
# - Had issues with pyinstaller v4.8.0. Works with 5.1.0
# - Issues with loading vtk modules addressed based on https://github.com/pyvista/pyvista-support/issues/167

block_cipher = None

# Determine the location of site packages as pyinstaller has trouble finding some modules
import dash
from pathlib import Path
sitepackages = Path(dash.__path__[0]).parent

# Make sure to add all of the necessary supporting data and packages
# that aren't automatically found py pyinstaller
added_files = [
    ('dashboard/database_*.pkl', 'dashboard'),
    ('dashboard/assets/*', 'dashboard/assets'),
    ('dashboard/static/examples/*', 'dashboard/static/examples'),
    ('unitcell/geometry/definitions/*', 'unitcell/geometry/definitions'),
    # ((sitepackages / Path('dash')).as_posix(), 'dash'),
    # ((sitepackages / Path('dash_bootstrap_components')).as_posix(), 'dash_bootstrap_components'),
    ((sitepackages / Path('dash_core_components')).as_posix(), 'dash_core_components'),
    ((sitepackages / Path('dash_html_components')).as_posix(), 'dash_html_components'),
    ((sitepackages / Path('dash_table')).as_posix(), 'dash_table'),
    ((sitepackages / Path('dash_vtk')).as_posix(), 'dash_vtk'),
    # ((sitepackages / Path('waitress')).as_posix(), 'waitress'),
]

# @TODO: The curren binary addition of blosc2 for tables is a fragile patch for Windows systems.
# This implementation likely won't work for *nix based systems. It also requires you to have
# a virtual environment setup in .venv. This needs to be made more robust in the future.
a = Analysis(['unitcellapp.py'],
             pathex=[],
             binaries=[(Path(".venv/bin/libblosc2.dll").as_posix(), "tables")],
             datas=added_files,
             hiddenimports=['cftime', 'cftime._strptime',
                            'vtkmodules', 'vtkmodules.all', 
                            'vtkmodules.util.numpy_support', 
                            'vtkmodules.numpy_interface', 
                            'vtkmodules.numpy_interface.dataset_adapter',
                            'dash', 'dash_bootstrap_components',
                            'waitress', 'tables'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='unitcellapp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='unitcellapp',
)
