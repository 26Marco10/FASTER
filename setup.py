from setuptools import setup, find_packages

setup(
    name="faster",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'nltk',
        'scikit-learn',
        'transformers',
        'torch',
        'reportlab',
        'psutil',
        'matplotlib',
        'tabulate',
        'evaluate',
        'reportlab',
        'sentencepiece',
        'accelerate>=0.26.0'
    ],
    entry_points={
        'console_scripts': [
            'faster=faster.cli:main'
        ]
    },
    python_requires='>=3.8',
)