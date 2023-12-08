import setuptools

from pathlib import Path

INSTALL_REQUIRES = ['numpy>=1.20.0', 'scipy>=1.10.1', 'opencv-python-headless>=4.1.1']

def get_version():
    locals_ = dict()

    with open(Path(__file__).parent / 'ecgmentations' / '__version__.py') as f:
        exec(f.read(), globals(), locals_)
        return locals_['__version__']

def get_long_description():
    with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
        return f.read()

setuptools.setup(
    name='ecgmentations',
    version=get_version(),
    description='Ecg augmentation library and easy to use wrapper around other libraries',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Rostislav Epifanov',
    author_email='rostepifanov@gmail.com',
    license='MIT',
    url='https://github.com/rostepifanov/ecgmentations',
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,
    extras_require={'tests': ['pytest']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
