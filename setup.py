from setuptools import setup

LICENSE = 'MIT'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
setup(
    name="ImageDataAugmentor",
    version="0.0.0",
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy>=1.18.1",
        "opencv-python>=4.1.2.30"
    ],
)
