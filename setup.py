# Always prefer setuptools over distutils
# To use a consistent encoding
from codecs import open
import os
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'monaco','readme.md'), encoding='utf-8') as f:
  long_description = f.read()


setup(
    name="monaco",

    version = "0.1",

    description='GPU accelerated collective Monte Carlo methods.',  # Required
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://www.anonymous.com/monaco',
    project_urls={
        'Bug Reports': 'https://github.com/anonymous/monaco/issues',
        'Source': 'https://github.com/anonymous/monaco',
    },
    author='X. Anonymous',
    author_email='anonymous@anonymous.com',

    python_requires='>=3',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: MIT License',

        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',

        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='monte carlo gpu',

    packages=[
        'monaco',
    ],

    package_data={
        'monaco': [
            'readme.md',
            'licence.txt',
            ]
    },

    install_requires=[
            'numpy',
            'torch',
    ],

    extras_require={
            'full': ['pykeops',
                     ],
            },
)

