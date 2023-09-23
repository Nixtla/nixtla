import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

dev = ["black", "holidays", "nbdev", "plotly", "python-dotenv", "statsforecast", "utilsforecast"]
distributed = ["dask", "fugue[ray]", "pyspark"]
plotting = ["utilsforecast[plotting]>=0.0.5"]

setuptools.setup(
    name="nixtlats",
    version="0.1.15",
    description="TimeGPT SDK",
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
    install_requires=["requests", "pandas", "httpx", "pydantic<2"],
    extras_require={
        "dev": dev + distributed + plotting,
        "distributed": distributed,
        "plotting": plotting,
    },
)
