import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

dev = [
    "black",
    "datasetsforecast",
    "nbdev",
    "plotly",
    "polars",
    "pre-commit",
    "python-dotenv",
    "pyreadr",
    "statsforecast",
    "neuralforecast",
    "hierarchicalforecast",
    "jupyterlab",
    "setuptools<70",
]
distributed = ["dask[dataframe]", "dask-sql", "fugue[ray]>=0.8.7", "pyspark", "ray[serve-grpc]"]
plotting = ["utilsforecast[plotting]>=0.2.2"]
date_extras = ["holidays"]

setuptools.setup(
    name="nixtla",
    version="0.5.2",
    description="Python SDK for Nixtla API (TimeGPT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nixtla/nixtla",
    packages=setuptools.find_packages(exclude=["action_files"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "httpx",
        "pandas",
        "pydantic",
        "requests",
        "tenacity",
        "tqdm",
        "utilsforecast>=0.2.2",
    ],
    extras_require={
        "dev": dev + distributed + plotting + date_extras,
        "distributed": distributed,
        "plotting": plotting,
        "date_extras": date_extras,
    },
)
