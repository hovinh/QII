# Adapted from: https://dzone.com/articles/executable-package-pip-install

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qii_tool",
    version='0.1.1',
    author="Ho Xuan Vinh",
    author_email="hxvinh.hcmus@gmail.com",
    description="A package implements Quantitative Input Influence method",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    url="https://github.com/hovinh/QII",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)