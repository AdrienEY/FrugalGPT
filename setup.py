from setuptools import setup, find_packages
import os

# Fonction pour récupérer tous les fichiers de configuration au niveau de la racine
def find_config_files():
    config_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith((".yml", ".yaml", ".json", ".ini", ".conf", ".cfg")):  # Ajoutez ici les extensions des fichiers config
                config_files.append(os.path.relpath(os.path.join(root, file)))
    return config_files

setup(
    name='FrugalGPT',
    version='0.0.1',
    author='Lingjiao Chen, Matei Zaharia, and James Zou',
    author_email='lingjiao@stanford.edu',
    description='The FrugalGPT library',
    # Recherche des packages dans 'src' uniquement
    packages=find_packages(where="src"),  
    package_dir={"": "src"},  # Déclare que le code source est dans 'src'
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
    # Inclure les fichiers de config dans le package data
    package_data={
        '': find_config_files(),  # Inclut les fichiers de configuration trouvés
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
