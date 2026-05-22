from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def read_requirements(path: str = "requirements.txt") -> list[str]:
    """Load non-comment, non-optional lines from requirements.txt."""
    req_path = ROOT / path
    lines: list[str] = []
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            continue
        lines.append(line)
    return lines


setup(
    name="pdmodel",
    version="0.0.2",
    author="leolim3092",
    author_email="your@email.com",
    description="An AI tool for screening Parkinson's disease",
    packages=find_packages(
        where=".",
        include=["pdmodel", "pdmodel.*"],
        exclude=["pdmodel.configs", "pdmodel.configs.*"],
    ),
    package_dir={"": "."},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "shap": ["shap>=0.41.0"],
        "optional": read_requirements("requirements-optional.txt"),
    },
)
