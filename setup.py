from setuptools import find_packages, setup
from typing import List

EXCLUDE_XTER = "-e ."
def get_requirements(file_path:str)->List[str]:
    """
    this function will return the list of requirements
    """
    requirements =[]
    with  open(file_path) as file_obj:
        requirements=file_obj.readlines()
        #strip any trainling newline characters from each line
        requirements=[req.strip() for req in requirements]
        # strip the "-e ."xter
        if EXCLUDE_XTER in requirements:
            requirements.remove(EXCLUDE_XTER)
    return requirements

setup(
    name ='mlproject',
    version='0.0.1',
    author='christian',
    author_email='krizlugtech@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    license='MIT',
    extras_require={
        'dev' : [
            'pytest',
            'flake8'
        ],
    }, # Additional development dependencies
    classifiers=[
        'programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
