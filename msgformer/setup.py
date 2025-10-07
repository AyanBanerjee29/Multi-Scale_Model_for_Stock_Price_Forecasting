from setuptools import setup, find_packages

setup(
    name="msgformer",
    version="1.0.0",
    description="MSGformer: Multi-Scale Graph-Transformer for Financial Forecasting",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "torch>=2.0.1",
        "yfinance>=0.2.28",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.1"
    ],
    python_requires=">=3.9",
)

