from setuptools import setup, find_packages

setup(
    name='FrugalGPT',
    version='0.0.1',
    author='Lingjiao Chen, Matei Zaharia, and James Zou',
    author_email='lingjiao@stanford.edu',
    description='The FrugalGPT library',
    packages=find_packages(where="src"),  # Recherche les packages sous src
    package_dir={"": "src"},  # Déclare que le code source est dans le répertoire 'src'
    install_requires=[
        'numpy',
        'cohere',
        'smart-open',
        'jsonlines',
        'anthropic',
        'scikit-learn',
        'evaluate',
        'scipy',
        'pandas',
        'sqlitedict',
        'torch',
        'transformers',
        'accelerate',
        'ai21',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
