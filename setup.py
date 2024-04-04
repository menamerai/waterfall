from setuptools import find_packages, setup

setup(
    name="waterfall",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
