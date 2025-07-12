#!/usr/bin/env python3
"""
Setup script for pix2pix GAN project
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="pix2pix-gan",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Complete pix2pix GAN implementation for Image-to-Image Translation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pix2pix-gan",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "visualization": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pix2pix-train=main:main",
            "pix2pix-inference=inference:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pix2pix-gan/issues",
        "Source": "https://github.com/yourusername/pix2pix-gan",
        "Documentation": "https://github.com/yourusername/pix2pix-gan/blob/main/README.md",
    },
    keywords="pix2pix gan deep-learning image-translation tensorflow generative-adversarial-networks",
    include_package_data=True,
    zip_safe=False,
)