from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
readme = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name="chae",
    version="0.2.1",
    description="Causal Hybrid Adaptive Engine",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="artnet82",
    url="https://github.com/artnet82/Chae-engine",
    license="MIT",

    packages=find_packages(
        include=["chae", "chae.*"],
        exclude=["tests", "tests.*", "examples", "examples.*", "benchmarks", "benchmarks.*"],
    ),
    include_package_data=True,
    zip_safe=False,

    python_requires=">=3.9",           # синхронизировано с вашим pyproject.toml
    install_requires=[
        "numpy>=1.24",
        "networkx>=3.1",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0"],
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)