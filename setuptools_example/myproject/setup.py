from setuptools import setup, find_packages

setup(
    name='coffeemaster',
    version='0.0.1',
    packages=find_packages(),
    install_requires = [
	    'torch',
	    'pandas>=1.4.2'
    ],
    package_data={'examplecoffee': ['tallSizeAmericano.txt']}
)
 