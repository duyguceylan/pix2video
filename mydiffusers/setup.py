#!/usr/bin/python

from __future__ import print_function
import os
import sys
import re
import os.path as op
from setuptools import find_packages, setup

## change directory to this module path
#try:
#    this_file = __file__
#except NameError:
#    this_file = sys.argv[0]
#this_file = os.path.abspath(this_file)
#if op.dirname(this_file):
#    os.chdir(op.dirname(this_file))
#script_dir = os.getcwd()

def find_version(file_path: str) -> str:
    version_file = open(file_path).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if not version_match:
        raise RuntimeError(f"Unable to find version string in {file_path}")
    return version_match.group(1)

def load_requirements(filename: str):
    with open(filename) as f:
        return [x.strip() for x in f.readlines() if "-r" != x[0:2]]

setup(name='mydiffusers', 

    version=find_version("mydiffusers/__init__.py"), 
    packages=find_packages(),
)