from setuptools import setup
import versioneer


setup(
    name='erebos',
    description='Model for converting GOES images to ground irradiance',
    author='Antonio Lorenzo',
    author_email='atlorenzo@email.arizona.edu',
    license='MIT',
    packages=['erebos'],
    zip_safe=True,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)
