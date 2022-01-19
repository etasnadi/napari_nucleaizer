#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()


# Add your dependencies in requirements.txt
# Note: you can add test-specific requirements in tox.ini
requirements = []
with open('requirements.txt') as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

def no_bullshit_pls(version):
    return ""

# https://github.com/pypa/setuptools_scm
use_scm = {
    "write_to": "napari_nucleaizer/_version.py",
    "version_scheme": no_bullshit_pls,
}

setup(
    name='napari_nucleaizer',
    author='Ervin Tasnadi',
    author_email='tasnadi.ervin@brc.hu',
    license='BSD-3',
    url='https://github.com/etasnadi/napari_nucleaizer',
    description='Napari integration of the nucleaizer algorithm. (https://nucleaizer.org)',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=requirements,
    version='0.1.10',
    setup_requires=['setuptools_scm'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Framework :: napari',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
    ],
    entry_points={'napari.plugin': 'napari_nucleaizer = napari_nucleaizer'},
)
