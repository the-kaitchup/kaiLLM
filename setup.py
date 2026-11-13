"""The payload contains Linux x86_64 native libraries for CPython 3.12."""

from setuptools import Distribution, setup


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=NativeDistribution)
