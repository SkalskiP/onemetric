import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="onemetric",
    version='0.0.1',
    python_requires=">=3.6",
    description="Metrics Library to Evaluate Machine Learning Algorithms in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/SkalskiP/onemetric",
    author="Piotr Skalski",
    author_email="piotr.skalski92@gmail.com",
    license='BSD',
    packages=setuptools.find_packages(exclude=('tests',)),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    extras_require={
        'tests': ['pytest']
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
