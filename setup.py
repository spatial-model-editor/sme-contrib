from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="sme_contrib",
    version="0.0.1",
    author="Liam Keegan",
    author_email="liam@keegan.ch",
    description="Useful modules for use with sme (Spatial Model Editor)",
    long_description=long_description,
    url="http://github.com/spatial-model-editor/sme_contrib",
    classifiers=[
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    license="MIT",
    packages=["sme_contrib"],
    install_requires=[
        "numpy",
    ],
    zip_safe=False,
)
