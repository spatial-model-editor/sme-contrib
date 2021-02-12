from setuptools import find_packages, setup

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sme_contrib",
    version="0.0.11",
    author="Liam Keegan",
    author_email="liam@keegan.ch",
    description="Useful modules for use with sme (Spatial Model Editor)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://sme_contrib.readthedocs.io/",
    project_urls={
        "Github": "https://github.com/spatial-model-editor/sme_contrib",
        "Issues": "https://github.com/spatial-model-editor/sme_contrib/issues",
    },
    classifiers=[
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    license="MIT",
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=requirements,
    zip_safe=False,
)
