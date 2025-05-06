## [Unreleased]
### Added
- requirements.txt for easier installation of dependencies
- updated the packages that can be installed by pypi (from rdkit 2020.03.1 to rdkit-pypi==2021.3.5)
- Installing python3.7 using ppa:deadsnakes/ppa repository

### Fixed
- Resolved `AttributeError: type object 'object' has no attribute 'dtype'` by removing `columns` prameter in `etoxpred_predict.py`.
- Resolved multiple dependencies issues.

### Changed
- Updated `README.md` to include instructions for setting up a virtual environment with Python 3.7 and installing dependencies.
