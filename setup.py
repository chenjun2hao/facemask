import platform
import sys
from io import open  # for Python 2 (identical to builtin in Python 3)
from setuptools import  find_packages, setup, dist

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

install_requires = [
    'numpy>=1.11.1', 'torch>=1.1.0', 'easydict', 'opencv-python'
]

def get_version():
    version_file = 'facemask/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

setup(
    name='facemask',
    version=get_version(),
    description='Face Mask detection',
    long_description=readme(),
    long_description_content_type='text/markdown',
    keywords='computer vision',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    license='MIT',
    url='https://github.com/chenjun2hao/facemask',
    author='Jun Chen',
    author_email='778961303@qq.com',
    install_requires=install_requires,
    zip_safe=False,
    data_files=['README.md']
    )
