from setuptools import setup, find_packages

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="mlax_nn",
    version="0.2.5",
    description="A pure functional machine learning library build on top of Google JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["mlax", "mlax.*"]),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    install_requires = [
        "jax>=0.4.8",
        "jaxlib>=0.4.7"
    ],
    extras_require = {
        "dev" : [
            "pytest==7.3.1"
        ],
    },
    url="https://github.com/zongyf02/mlax",
    author="Yifan Zong",
    author_email="y22zong@uwaterloo.ca"
)
