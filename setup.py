#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Randall Balestriero"
import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='symjax',  

     version='0.1',

     scripts=['pkg'] ,

     author="Randall Balestriero",

     author_email="randallbalestriero@gmail.com",

     description="A symbolic JAX software",

     long_description=long_description,

   long_description_content_type="text/markdown",

     url="https://github.com/RandallBalestriero/SymJAX.git",

     packages=setuptools.find_packages(),

     classifiers=[
         "Natural Language :: English",

         "Development Status :: 3 - Alpha",

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: Unix",

     ],

 )
