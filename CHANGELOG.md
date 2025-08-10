# Changelog

All notable changes to this project will be documented in this file.  
This project adheres to [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.2.1] - 2025-08-10

### Added
- **Signal-based Weighting** – Introduced `use_signal_weighting` parameter in reduction pipeline for improved signal-to-noise ratio in contrast estimation by weighting pixels based on expected signal strength.
- **Progress Bar Control** – Added `use_progress_bar` parameter to enable/disable progress feedback during long-running reductions, improving user experience and transparency.

### Changed
- **Reduction Mask Default** – Changed default `reduction_mask_size_in_lambda_over_d` from 2.0 to 1.0 pixels for better performance in typical science cases.
- **Search Region Expansion** – Increased default `search_region_outer_bound` from 55 to 85 pixels to improve detection performance.
- **Import Organization** – Switched from relative to absolute imports in regression module for better maintainability.

### Fixed
- **Candidate Validation** – Improved robustness in template matching and detection pipeline when no candidates survive second iteration, preventing downstream errors.

---

## [1.2.0] - 2025-07-07

### Added
- **Config Parameter System** – Introduced a new parameter configuration system based on dataclasses, following the spherical pipeline approach, improving flexibility and clarity in setting up defaults. Legacy objects are still used under the hood for now.

### Changed
- **Search Radius Consistency** – Standardized default `iterative_search_exclusion_radius` across detection pipeline and spectral extraction to 15 pixels, adjustable by users.
- **Annulus Width Consistency** – Removed hardcoded annulus width in spectral extraction to match pipeline-level configuration.
- **Improved Server Handling** – Ensured Ray server properly shuts down even in case of unexpected crashes, increasing pipeline robustness.
- **Outer Search Region Bound** – Cast `outer_search_region_bound` explicitly to integer for improved type safety.
- **Notebook Progressbar** – Notebook compatible progressbar is used by all modules. 

### Fixed
- **Critical FWHM Bug** – Corrected calculation of PSF size (`lambda/D` conversion) compatible with astropy >= 6.1. This critical bug kept the pipeline from working properly with up-to-date astropy.
- **Pickling Consistency** – Standardized object serialization using dill in both reduction and detection modules, fixing parameter persistence issues.
- **Latex Syntax Warning** – Fixed incorrect LaTeX escape sequences in detection curve plotting to eliminate syntax warnings.

---

## [1.1.0] - 2025-04-13

### Added
- **Forced Photometry Mode** – Introduced an option to perform *forced photometry* for known companion positions, allowing direct flux measurement at specified coordinates.  
- **Pickle I/O Utilities** – Added utility functions in `trap.utils` for saving and loading TRAP objects (e.g. results or models) via pickle, simplifying persistence of analysis results.  
- **Example Data & Tutorial** – Provided a Jupyter tutorial notebook and sample dataset in the `examples/` directory to help users get started with TRAP’s workflow.  
- **Documentation & CI** – Established a Sphinx documentation framework (`docs/` directory) and added continuous integration workflows (GitHub Actions for testing and docs).  
- **GitHub Templates** – Added issue templates for bug reports and feature requests, and a pull request template.

### Changed
- **Package Layout** – Restructured the project to a modern *“src”* layout under `src/trap/`, using PEP 621-based `pyproject.toml`.  
- **Python & Dependency Support** – Now supports Python 3.11+ (including Python 3.12). Dropped support for Python 3.9/3.10.  
- **Detection Defaults** – Improved defaults in the detection pipeline for better performance and usability.  
- **Detection Map Normalization** – Detection maps are now automatically **empirically normalized** to correspond to the detection significance (in σ) of a point source, improving interpretability.  
- **Logging Verbosity** – Reduced Ray's logging and multiprocessing noise for a cleaner CLI experience.  
- **Cross-Validation** – Adjusted the regression cross-validation strategy for better model selection.

### Fixed
- **Species Template Matching** – Fixed bugs in spectral template matching with `species`.  
- **NaN and Zero Handling** – Improved robustness to missing data and zero placeholders.  
- **Result Saving** – Fixed issues with saving contrast curves and spectral extraction overwriting detection maps.  
- **Parameter Bugs** – Fixed argument handling in wrappers and detection masking logic.  
- **Miscellaneous Fixes** – Code cleanup, better NaN handling, and bug fixes across modules.

### Removed
- **Legacy Code** – Removed unused code paths, imports, and debug routines.  
- **Python 3.9/3.10 Support** – Dropped support for older Python versions due to updated dependencies.

---

## [1.0.0] - 2024-03-28

### Added
- Initial release.

### Changed
- Initial implementation of core functionality.

### Fixed
- No known issues.

[Unreleased]: https://github.com/m-samland/trap/compare/v1.2.1...HEAD 
[1.2.1]: https://github.com/m-samland/trap/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/m-samland/trap/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/m-samland/trap/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/m-samland/trap/releases/tag/v1.0.0