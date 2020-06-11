#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Randall Balestriero"

from setuptools import setup, find_packages

with open("README.md", "r") as fh:

    long_description = fh.read()

setup(

     name='symjax',  

     version='0.3.1',

     author="Randall Balestriero",

     author_email="randallbalestriero@gmail.com",

     description="A Symbolic JAX software",

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://github.com/RandallBalestriero/SymJAX.git",

     packages=find_packages(exclude=["examples"]),

     classifiers=[
         "Natural Language :: English",

         "Development Status :: 3 - Alpha",

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: Unix",

     ],
     python_requires='>=3.6',
 )
