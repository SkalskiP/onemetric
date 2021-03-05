import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="cv-metrics",
    version='0.0.1',
    description="Metrics To Evaluate Computer Vision Algorithms in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/SkalskiP/cv-metrics",
    author="Piotr Skalski",
    author_email="piotr.skalski92@gmail.com",
    license='BSD',
    packages=setuptools.find_packages(),
    include_package_data=True
)
