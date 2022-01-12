from setuptools import find_packages, setup

setup(
    name='rgs_lib',
    packages=find_packages(),
    version='0.1.0',
    description='Ridge Regression for SAE',
    author='Rifki Gustiawan',
    license='MIT',
    install_requires=['random','pandas','matplotlib','scipy', 'sklearn'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='test',
)
