from setuptools import setup, find_packages

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="mlax_nn",
    version="0.0.2",
    description="A pure functional machine learning library build on top of Google JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["mlax", "mlax.*"]),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires = [
        "jax>=0.3.23",
        "jaxlib>=0.3.22"
    ],
    extras_require = {
        "dev" : [
            # "pytest>=3.8"
        ],
    },
    url="https://github.com/zongyf02/mlax",
    author="Yifan Zong",
    author_email="y22zong@uwaterloo.ca"
)