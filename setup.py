import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='causalbenchmark',
    version='0.1.0',
    author='Hendrik Plett',
    author_email='hendrik.plett@web.de',
    packages=['src.causalbenchmark', 'src.causalbenchmark.compute', 'src.causalbenchmark.visualize'],
    scripts=[],
    url='https://github.com/HendrikPlett/MA',
    description='Python implementation of the GES algorithm for causal discovery',
    install_requires=requirements
)