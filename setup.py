# -*- coding: utf-8 -*-

import setuptools
import versioneer

short_description = "Optimizing numpys einsum function"
try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except: # noqa
    long_description = short_description


if __name__ == "__main__":
    setuptools.setup(
        name='opt_einsum',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description=short_description,
        author='Daniel Smith',
        author_email='dgasmith@icloud.com',
        url="https://github.com/dgasmith/opt_einsum",
        license='MIT',
        packages=setuptools.find_packages(),
        python_requires='>=3.6',
        install_requires=[
            'numpy>=1.7',
        ],
        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
        ],

        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        zip_safe=True,
        long_description=long_description,
        long_description_content_type="text/markdown"
    )
