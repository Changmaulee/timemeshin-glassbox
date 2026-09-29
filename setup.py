from setuptools import setup, find_packages

setup(
    name="timemeshin-glassbox",
    version="1.0.0",
    author="Chandra Mouli",
    description="A 6-Layer Inherently Interpretable Hierarchical State Space Language Model Architecture",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pyyaml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
