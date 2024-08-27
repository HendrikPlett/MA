from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='causalbenchmark',
    version='0.1.0',
    author='Hendrik Plett',
    author_email='hendrik.plett@web.de',
    description='Benchmarking causal discovery algorithms on the edge level.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    url='https://github.com/HendrikPlett/MA',
    install_requires=requirements,
    python_requires='>=3.6'
)