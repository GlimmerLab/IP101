from setuptools import setup, find_packages
import os

# 读取README.md作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="image_processing",
    version="1.0.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="高性能图像处理库",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-processing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "image_processing": [
            "*.so",
            "*.dll",
            "*.dylib",
        ],
    },
    entry_points={
        "console_scripts": [
            "image_processing=image_processing.cli:main",
        ],
    },
)