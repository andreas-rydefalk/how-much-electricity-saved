# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.1] - 2023-04-09

### Fixed

- Fixed position of the highlighted baseline period on the monthly bar chart


## [0.2.0] - 2023-04-09

### Added

- Improved the monthly bar chart
  - Highlighted the baseline period
  - Added average monthly outside temperature as a line
  - Added the diff in kHw compared to baseline prediction
  - Added the new columns of `self.monthly_df` to the tests
  - Added more sample data

### Changed

- Simplified and updated the example and the example output png.

## [0.1.0] - 2023-01-22

### Added

- Added a function for creating a monthly bar chart graph of the electricity consumption
- Added some tests

### Changed

- Changed the required version of Python from 3.6 to 3.9 in setup.cfg, because I have only tested this with 3.9 and 3.11. I'll not make any promises for earlier versions.

## [0.0.1] - 2022-12-30

### Added

- This CHANGELOG file
- Making this an installable package with 
  - py_project.toml
  - setup.cfg
  - setup.py
- .vscode/settings.json
- Working PoC in the src/how_much_electricity_saved
- Example sample data, python script and output png in the examples folder
- Basic instructions in README.md
