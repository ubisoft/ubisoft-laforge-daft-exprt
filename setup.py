#!/usr/bin/env python
import os

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='daft_exprt',
    author='Julian Zaidi',
    author_email='julian.zaidi@ubisoft.com',
    description='Package for training and generating speech representations with Daft-Exprt acoustic model.',
    url='https://github.com/ubisoft/ubisoft-laforge-daft-exprt',
    license='© [2021] Ubisoft Entertainment. All Rights Reserved',
    long_description=readme,
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Operating System :: Linux'
    ],
    setup_requires=['setuptools_scm'],
    python_requires='>=3.8',
    install_requires=open(os.path.join('environment', 'pip_requirements.txt')).readlines(),
    extras_require={
        'pytorch': ['torch==1.9.0+cu111', 'torchaudio==0.9.0', 'tensorboard']
    },
    package_dir={'':'src'},
    packages=find_packages('src'),
    use_scm_version={
        'root': '.',
        'relative_to': __file__,
        'version_scheme': 'post-release',
        'local_scheme': 'dirty-tag'
    }
)
