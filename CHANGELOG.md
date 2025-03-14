# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
The rules for this file:
  * entries are sorted newest-first.
  * summarize sets of changes - don't reproduce every git log comment here.
  * don't ever delete anything.
  * keep the format consistent:
    * do not use tabs but use spaces for formatting
    * 79 char width
    * YYYY-MM-DD date format (following ISO 8601)
  * accompany each entry with github issue/PR number (Issue #xyz)
-->
## [0.3.0] - 2024-03-14

### Added in O.3.0

- Notebook for benchmarking Metrics : Neq,RMSF,Menger Curvature
- Notebooks for generating Figures for paper
- Additionnal Paper figure in pdf

### Changed in 0.3.0

- changed order of test in Menger class :  spacing -> selection => selection -> spacing
- Correct typo in documentation where argument select was wrongly refered as selection

## [0.2.0] - 2024-03-04

### Added in 0.2.0

- Better error handling for spacing and selection parameters
- Improved documentation with usage examples

## [0.1.0] - 2024-02-28

### Added in 0.1.0

- Initial release
- MengerCurvature analysis class
- Parallel computation support
- Support for MDAnalysis trajectories
- Unit test suite with code coverage

### Dependencies

- NumPy
- MDAnalysis
- Numba
