import setuptools

INSTALL_REQUIRES = ['numpy>=1.24.4']

setuptools.setup(
    name='ecgmentations',
    version='0.0.0',
    description='',
    long_description=''
    author='Epifanov Rostislav',
    license='MIT',
    url='https://github.com/rostepifanov/egcmentations',
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,
    extras_require={},
    classifiers=[]
)
