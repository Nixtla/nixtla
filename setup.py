import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

dev = ["black", "nbdev", "plotly", "python-dotenv", "openbb", "statsforecast", "utilsforecast"]
distributed = ["dask[complete]", "fugue[ray]", "pyspark"]

setuptools.setup(
    name="nixtlats",
    version="0.1.14",
    description="Nixtla SDK",
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
        "dev": dev + distributed,
        "distributed": distributed,
    },
)
