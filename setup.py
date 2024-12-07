from setuptools import setup, find_namespace_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A Python package for training and using an XGBoost model."

setup(
    name="xgboost_package",
    version="1.0.0",
    author="CVK0406",
    author_email="21522236@gm.uit.edu.vn",
    description="A Python package for training and using an XGBoost model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CVK0406/IS353.P12_Nhom4",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "pandas>=2.1.1",
        "xgboost>=1.7.6",
        "scikit-learn>=1.3.1",
        "numpy>=1.24.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.2",
        "lightgbm>=3.3.5",
        "joblib>=1.3.2",
        "networkx>=3.1",
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "visualization": ["matplotlib", "seaborn"],
        "lightgbm": ["lightgbm"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
)