import setuptools

INSTALL_REQUIRES = ['numpy>=1.20.0', 'opencv-python-headless>=4.1.1']

setuptools.setup(
    name='ecgmentations',
    version='0.0.3',
    description='',
    long_description='',
    author='Epifanov Rostislav',
    license='MIT',
    url='https://github.com/rostepifanov/egcmentations',
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,
    extras_require={},
    classifiers=[]
)
