from setuptools import setup, find_packages

setup(
    name='fewshot_prp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tqdm>=4.56.0',
        'torch>=2.1.2',
        'python-terrier>=0.10.0',
        'requests>=2.31.0',
        'urllib3>=2.2.1',
        'transformers>=4.40.1',
        'pandas>=2.2.2'
    ],
    entry_points={
    "concole_scripts": [
        "fewshot_prp = fewshot_prp:main"
    ]
    }
)