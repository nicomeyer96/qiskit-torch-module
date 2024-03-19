import setuptools

with open('README.md', 'r') as ff:
    long_description = ff.read()

setuptools.setup(
    name='qiskit-torch-module',
    version='1.0',
    description='Qiskit-Torch-Module: Fast Prototyping of Quantum Neural Networks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Fraunhofer IIS',
    license='Apache License 2.0',
    platforms='any',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
    ],
    packages=setuptools.find_packages(),
    python_requires='~=3.12',
    install_requires=[
        'qiskit~=1.0.0',  # backward compatible up to qiskit v0.44.0
        'qiskit-algorithms~=0.3.0',
        'torch~=2.2.1',
        'threadpoolctl~=3.3.0',
    ],
    include_package_data=True
)
