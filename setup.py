from setuptools import setup, find_packages
import versioneer


setup(
    name="erebos",
    description="Model for converting GOES images to ground irradiance",
    author="Antonio Lorenzo",
    author_email="atlorenzo@email.arizona.edu",
    license="MIT",
    packages=find_packages(),
    zip_safe=True,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    entry_points={"console_scripts": ["erebos=erebos.cli:cli"]},
)
