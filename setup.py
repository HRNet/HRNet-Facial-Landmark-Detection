## setup.py
from setuptools import setup

setup(
    name='hrnet_face_landmark',
    version='0.1',
    packages=['hrnet_face_landmark'],
    # package_dir={'hrnet_face_landmark':''},
    install_requires=[
        'torch>=1.0.0',
    ]
)