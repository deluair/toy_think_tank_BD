from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="newsguard-bangladesh",
    version="0.1.0",
    author="NewsGuard Research Team",
    author_email="research@newsguard.com",
    description="Agent-based simulation framework for Bangladesh's digital news ecosystem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/newsguard/bangladesh-simulation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Sociology",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.9.0",
            "torch-gpu>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "newsguard-sim=newsguard.cli:main",
            "newsguard-dashboard=newsguard.dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "newsguard": [
            "data/templates/*.json",
            "data/demographics/*.csv",
            "config/*.yaml",
        ],
    },
}