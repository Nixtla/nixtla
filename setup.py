import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

dev = [
    "black",
    "datasetsforecast",
    "fire",
    "hierarchicalforecast",
    "jupyterlab",
    "nbdev",
    "neuralforecast",
    "numpy<2",
    "plotly",
    "polars",
    "pre-commit",
    "pyreadr",
    "python-dotenv",
    "pyyaml",
    "setuptools<70",
    "statsforecast",
    "tabulate",
]
distributed = ["fugue[dask,ray,spark]>=0.8.7", "pandas<2.2", "ray<2.6.3"]
plotting = ["utilsforecast[plotting]>=0.2.7"]
date_extras = ["holidays"]

setuptools.setup(
    name="nixtla",
    version="0.6.2",
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
    python_requires=">=3.9",
    install_requires=[
        "annotated-types",
        "httpx",
        "orjson",
        "pandas",
        "tenacity",
        "tqdm",
        "utilsforecast>=0.2.3",
    ],
    extras_require={
        "dev": dev + plotting + date_extras,
        "distributed": distributed,
        "plotting": plotting,
        "date_extras": date_extras,
    },
)
