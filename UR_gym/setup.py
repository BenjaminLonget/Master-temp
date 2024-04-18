from setuptools import setup, find_packages

setup(
    name="UR_gym",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["gymnasium==0.29.1", "pybullet==3.2.6"],
)