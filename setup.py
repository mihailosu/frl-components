from setuptools import setup, find_packages

setup(
    name="frl_components",  
    version="0.1.0",
    packages=find_packages(),
    install_requires=[  
        'tensorflow>=2.0',
        'flwr>=1.12.0',
        'scikit-learn==1.7.1',
        'apricot-select>=0.6.1',
        'scikit-learn-extra==0.3.0'
    ],
    description="A custom model with an autoencoder and attention mechanism",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Mihailo Ilic",
    author_email="milic@dmi.uns.ac.rs",
    url="https://github.com/mihailosu/frl-components"
)
