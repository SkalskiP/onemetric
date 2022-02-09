import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="onemetric",
    version='0.1.2',
    python_requires=">=3.7",
    description="Metrics Library to Evaluate Machine Learning Algorithms in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/SkalskiP/onemetric",
    author="Piotr Skalski",
    author_email="piotr.skalski92@gmail.com",
    license='BSD',
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "Pillow",
        "numpy",
        "seaborn",
        "matplotlib",
        "dataclasses-json",
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-cov',
            'coverage',
        ]
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ]
)
