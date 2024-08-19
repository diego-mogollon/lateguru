#setup.py
from setuptools import setup
from setuptools import find_packages

#standard pytthon code for requirements.txt file
with open('requirement.txt') as f:
    content = f.readlines()
    requirements = [x.strip() for x in content]

#standard python code for packages logic
setup(name='lateguru', #package name
      description='predict flight delays', #optional description
      install_requires=requirements, #installs packages from requirements.txt
      packages=find_packages()) #contanstly looks for any new packages in our project, but our main is 'lateguru_ml'
