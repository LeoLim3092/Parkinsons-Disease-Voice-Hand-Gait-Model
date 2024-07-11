from setuptools import setup, find_packages

setup(
    name='pdmodel',
    version='0.0.2',
    author='leolim3092',
    author_email='your@email.com',
    description='An AI tool for screening Parkinson\'s disease',
    packages=find_packages(include=['pdmodel', 'pd_model.*']),  # Automatically discover and include all Python packages in your project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # List your package dependencies here
        'numpy',
        'matplotlib',
    ],
)
