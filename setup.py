from setuptools import setup, find_packages

setup(
    name="srep_simulator",
    packages=find_packages(),
    entry_points={
        "console_scripts": ['srep-simulator=simulator.main:main']
    },
    author='Novak Boskov',
    author_email='boskov@bu.edu',
    install_requires=[
        'networkx',
        'numpy',
        'tqdm',
        'pytest',
        'matplotlib',
        'pandas',
        'numpy',
        'scipy',
        'statsmodels',
        'seaborn',
        'fitter'
    ]
)
