from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function will return the list of requirements
    """
    requirement_lst: List[str] = []
    try:
        with open('requirements.txt') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("Requirements.txt file not found.")
    return requirement_lst

print(get_requirements())

setup(
    name='NetWorkSecurity',
    version='0.0.1',
    author='Atharva Rai',
    author_email='atharvarai07@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
    license='MIT',
)
