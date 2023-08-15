import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nixtlats",
    version="0.1.6",
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
    install_requires=["requests", "pandas"],
    extras_require={"dev": ["nbdev", "plotly", "python-dotenv", "openbb", "statsforecast"]},
)
