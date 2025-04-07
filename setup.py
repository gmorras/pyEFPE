import sys, os
from setuptools import setup, find_packages

#ensure Cython is installed before importing it
try:
    from Cython.Build import cythonize
except ImportError:
    print("Cython not found, installing it now...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "Cython"], check=True, capture_output=True)
    from Cython.Build import cythonize

setup(
    name             = 'pyEFPE',
    description      = 'Python package for an efficient fully precessing eccentric inspiral model.',
    version          = '0.0',
    author           = 'Gonzalo Morrás, Geraint Pratten, Patricia Schmidt',
    author_email     = 'gonzalo.morras@ligo.org',
    packages         = find_packages(),
    package_dir      = {'pyEFPE': 'pyEFPE'},
    ext_modules      = cythonize("pyEFPE/utils/cython_utils.pyx"),
    url              = '',
    download_url     = '',
    install_requires = ['numpy','scipy>=1.9.3','Cython'],
)

