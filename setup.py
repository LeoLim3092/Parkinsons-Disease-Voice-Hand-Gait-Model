from setuptools import setup, find_packages

setup(
    name='pd_model',
    version='0.1',
    author='leolim3092',
    author_email='your@email.com',
    description='An AI tool for screening Parkinson\'s disease',
    packages=find_packages(include=['pdmodel', 'pd_model.*']),  # Automatically discover and include all Python packages in your project
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: Apache License 2.0',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        # List your package dependencies here
        'numpy',
        'matplotlib',
    ],
)
