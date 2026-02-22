# Changelog

All notable changes to this project will be documented in this file.

## [1.1.2] - 2026-02-22
### Features
- feat: Display accuracy evaluation based on wave ranges (Layer 2 & 4) in device metrics. ([#3a29b3a](https://github.com/user/repo/commit/3a29b3ae2974995cf670928c39985ed8e3292201))
- feat: Implement device performance evaluation with metrics for data continuity, noise, and resolution, displayed in a new UI tab. ([#770409f](https://github.com/user/repo/commit/770409f01335626df154dd0fc12995e9edd71305))
- feat: Remove backfilling of initial NaN values for permutation entropy and rolling variance, and instead visually highlight these NaN regions in Plotly charts. ([#a54e248](https://github.com/user/repo/commit/a54e248c72fbcfc0a51a1f3f1850f0daed4e49db))
- feat: Add and display country information for location data ([#334f0ec](https://github.com/user/repo/commit/334f0ece60901b1a12c002ab351d1e5e2ceb97f8))

### Bug Fixes
- fix: Backfill permutation entropy and rolling variance to prevent Plotly rendering gaps and separate raw and smoothed dP/dt into distinct charts. ([#23bc73f](https://github.com/user/repo/commit/23bc73fcd4a1091860b6f0348ca6f644db4f0c94))

### Refactor
- refactor: Set Plotly charts to SVG render mode and optimize min/max point visualization for SVG. ([#7e5f7f0](https://github.com/user/repo/commit/7e5f7f092422e776e765a29a8fa8562e07db341b))

## [1.1.1] - 2026-02-22
feat: support country in location.csv and update UI

## [1.1.0] - 2026-02-22
### Features
- feat: Implement loading and utilization of location data for display and tidal analysis. ([#1b40452](https://github.com/user/repo/commit/1b404521d3369c03cd428ee4c026a8ea182b76a9))

## [1.0.2] - 2026-02-22
### Refactor
- Refactor: Categorize git commits into Conventional Commits sections with Markdown headings and links for improved changelog generation. ([#aa3f87b](https://github.com/user/repo/commit/aa3f87bf30aa1f1c7972f38be02e9940649e8ea8))

## [1.0.1] - 2026-02-22
- Minor updates and improvements.

## [1.0.0] - 2026-02-22
- Initial release of AetherBaro Pressure Analyzer.
- Implemented 5-layer physical analysis.
- Added support for LG V60 and Sony XZ2 hardware metadata.
- Integrated theoretical tidal models (Solar/Lunar).
- Implemented multipoint extremum detection.
