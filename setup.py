import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

dev = [
    "black",
    "datasetsforecast",
    "nbdev",
    "plotly",
    "pre-commit",
    "python-dotenv",
    "statsforecast",
]
distributed = ["dask[dataframe]", "fugue[ray]>=0.8.7", "pyspark", "ray[serve-grpc]"]
plotting = ["utilsforecast[plotting]>=0.0.5"]
date_extras = ["holidays"]

setuptools.setup(
    name="nixtlats",
    version="0.3.0",
    description="Python SDK for Nixtla API (TimeGPT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nixtla/nixtla",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "httpx",
        "pandas",
        "pydantic<2",
        "requests",
        "tenacity",
        "utilsforecast>=0.0.13",
    ],
    extras_require={
        "dev": dev + distributed + plotting + date_extras,
        "distributed": distributed,
        "plotting": plotting,
        "date_extras": date_extras,
    },
)
