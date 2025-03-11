from setuptools import setup

setup(
    name="ukb_deppred",
    version="0.1.0",
    python_requires=">=3.9, <4",
    install_requires=[
        "datalad ~= 1.1.2",
        "nipype ~= 1.8.5",
        "numpy ~= 2.0",
        "pandas ~= 2.2.2",
        "scipy ~= 1.15.0",
        "scikit-learn ~= 1.5.1",
        "tables ~= 3.10.0",
        "Pillow ~= 11.1.0",
        "semopy ~=2.3.11",
        "graphviz ~= 0.20.3",
    ],
    extras_require={
        "dev": [
            "flake8",
            "pyre-check",
            "pytest",
            "pytest-cov",
            "pandas",
        ],
    },
)