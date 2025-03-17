import os

from setuptools import setup


__version__ = "0.1.0"


# Read requirements from requirements.txt
def get_requirements():
    requirements = []
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


setup(
    name="scenariomax",
    version=__version__,
    packages=["scenariomax"],
    install_requires=get_requirements(),
    description="A toolkit for scenario-based autonomous vehicle testing",
    python_requires=">=3.10",
)
