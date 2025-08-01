import os

import setuptools

with open('../README.md', 'r') as fh:
    long_description = fh.read()

# Since nuScenes 2.0 the requirements are stored in separate files.
with open('requirements.txt') as f:
    req_paths = f.read().splitlines()
requirements = []
for req_path in req_paths:
    if req_path.startswith('#'):
        continue
    req_path = req_path.replace('-r ', '')
    with open(req_path) as f:
        requirements += f.read().splitlines()

setuptools.setup(
    name='nuscenes-devkit',
    version='1.1.11',
    author='Holger Caesar, Oscar Beijbom, Qiang Xu, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, '
           'Sergi Widjaja, Kiwoo Shin, Caglayan Dicle, Freddy Boulton, Whye Kit Fong, Asha Asvathaman, Lubing Zhou '
           'et al.',
    author_email='nuscenes@motional.com',
    description='The official devkit of the nuScenes dataset (www.nuscenes.org).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nutonomy/nuscenes-devkit',
    python_requires='>=3.6',
    install_requires=requirements,
    packages=setuptools.find_packages('../python-sdk'),
    package_dir={'': '../python-sdk'},
    package_data={'': ['*.json']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'License :: Free for non-commercial use'
    ],
    license='apache-2.0'
)
