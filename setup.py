from setuptools import setup, find_packages

setup(
    name="abacus2deepmd",
    version="0.3.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "pandas", "scipy", "scikit-learn", "dpdata"],
    entry_points={"console_scripts": ["abacus2deepmd=abacus2deepmd.main:main"]},
    python_requires=">=3.8",
)
