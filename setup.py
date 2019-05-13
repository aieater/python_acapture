from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))

version = '1.0.4'

install_requires = [
    'mss',
    'opencv-python',
    'pygame',
    'configparser',
]

readme = open("README.md").read()

setup(name='acapture',
    version=version,
    description="Async web camera/video/images/screenshot capturing library.",
    long_description="https://github.com/aieater/python_acapture\n\n"+readme,
    long_description_content_type='text/markdown',
    classifiers=(
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ),
    keywords='opencv python screenshot video mp4 capture async web camera image',
    author='Pegara, Inc.',
    author_email='info@pegara.com',
    url='https://github.com/aieater/python_acapture',
    license='MIT',
    packages=['acapture'],
    zip_safe=False,
    install_requires=install_requires,
    entry_points={}
)
