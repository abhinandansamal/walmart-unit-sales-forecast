import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "walmart-unit-sales-forecast"
AUTHOR_USER_NAME = "abhinandansamal"
SRC_REPO = "walmart-unit-sales-forecast"
AUTHOR_EMAIL = "samalabhinandan06@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package to forecast walmart sales data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "numpy",
        "cython",
        "pandas",
        "seaborn",
        "matplotlib",
        "statsmodels",
        "scipy",
        "scikit-learn",
        "plotly",
        "pyyaml",
        "prophet",
        "tqdm",
        "joblib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)