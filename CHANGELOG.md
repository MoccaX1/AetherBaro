# Changelog

All notable changes to this project will be documented in this file.

## [1.1.18] - 2026-02-27
- Minor updates and improvements.

## [1.1.17] - 2026-02-27
### Features
- feat: improve wave period marker styling in Plotly charts. ([#66d30be](https://github.com/user/repo/commit/66d30bee3a05a9bbc43fff5d7058dd853c20ea15))

## [1.1.16] - 2026-02-27
### Features
- feat: Add wave signal (P - Synoptic) plot, make CWT and HHT/EMD wave analysis optional ([#1ec3240](https://github.com/user/repo/commit/1ec324074f38daca0a733e77a292b611209ac183))

## [1.1.15] - 2026-02-27
### Features
- feat: Implement multi-method spectral analysis including FFT, PSD, STFT, CWT, and HHT for wave detection, along with new helper functions and diagnostic files. ([#6a6cbfd](https://github.com/user/repo/commit/6a6cbfd21d11ddd4232c326624481b515385919c))

## [1.1.14] - 2026-02-26
### Features
- feat: Add spectral detrending verification script and improve tidal model with S3 component and duration-aware wave search windows. ([#0448ebc](https://github.com/user/repo/commit/0448ebc4c394a05ae152b387c045fad3d94e7a90))

## [1.1.13] - 2026-02-25
### Features
- feat: Refine noise floor calculation and spectral decomposition for improved wave detection and reliability assessment, introducing new analysis scripts. ([#bdfe95c](https://github.com/user/repo/commit/bdfe95c895bd12914878f8555b28c76e9f65bb48))

## [1.1.12] - 2026-02-24
### Bug Fixes
- fix: streamline `st.dataframe` width handling in `app.py`. ([#f207ad0](https://github.com/user/repo/commit/f207ad004770175b0f565b20578737a5c8065018))

### Chores
- chore: add `skyfield` package ([#c5fc845](https://github.com/user/repo/commit/c5fc8451121d1a539d1692e4788e252e562939b5))

## [1.1.11] - 2026-02-24
### Features
- feat: Add new pressure sample datasets and related analysis scripts, updating gitignore. ([#55684ce](https://github.com/user/repo/commit/55684ce776506fa5d5ee573bc3e159f0f3aaa210))
- feat: Enhance noise analysis to distinguish white and pink noise, and introduce wave evaluation scripts. ([#30f9c95](https://github.com/user/repo/commit/30f9c95427211aef84817cbc22c9879e25419bf2))

## [1.1.10] - 2026-02-24
### Features
- feat: Enhance wave analysis reliability reporting by incorporating actual noise from data and add supporting analysis scripts for amplitude and FFT calculations. ([#c1a0b66](https://github.com/user/repo/commit/c1a0b66e294857f0c2ee3db6f0674f8e0046dafb))

## [1.1.9] - 2026-02-24
### Bug Fixes
- fix: Amplitude Calculation for Long Waves in Layer 2 ([#710e960](https://github.com/user/repo/commit/710e9609a942056d6d3a1acbf1a05376f56f0add))

## [1.1.8] - 2026-02-23
### Bug Fixes
- fix: Dynamically set data directory path for cross-platform compatibility and add an existence check with an error message. ([#0a7c46a](https://github.com/user/repo/commit/0a7c46a1e0e35a5cc7d4be990e51838169208964))

### Documentation
- docs: update README documentation ([#7175c9b](https://github.com/user/repo/commit/7175c9b8b30061fadd2092dba4b0b4082684d247))

### Chores
- chore: update project dependencies in requirements.txt ([#634019c](https://github.com/user/repo/commit/634019c9886db63b9ac8a72686e3cfdadc7bed52))

## [1.1.7] - 2026-02-23
### Features
- feat: Display total measurement duration in the overview and extend tide range block annotations to wave and entropy plots. ([#f251286](https://github.com/user/repo/commit/f2512867e014f7132e4dee31107714c0d41ece0b))
- feat: Add new pressure sample data files including location, time, device, and raw data, and update .gitignore to include the new directory. ([#6ba7bfc](https://github.com/user/repo/commit/6ba7bfc1e760345caa06061c2701483977e73408))

## [1.1.6] - 2026-02-22
### Features
- feat: Add descriptive help tooltips to various headers and metrics for enhanced user guidance. ([#d44af17](https://github.com/user/repo/commit/d44af1735ce9ae715d09972d27eae4121cb9cc9e))

## [1.1.5] - 2026-02-22
### Refactor
- refactor: Replace discrete peak annotations with continuous range highlighting for pressure tolerance zones. ([#295b972](https://github.com/user/repo/commit/295b972a1c8d6124c53ed5d018ebeb5ee6e23168))

### Others
- Merge branch 'main' of https://github.com/MoccaX1/Project ([#edbbcfb](https://github.com/user/repo/commit/edbbcfb9323c49164ed149bc94f2c1c61e727bd5))
- Added Dev Container Folder ([#e925f78](https://github.com/user/repo/commit/e925f78da3f3a5ef2e5c48517832eac30fa6dd37))

## [1.1.4] - 2026-02-22
### Features
- feat: Distinguish annotation methods for noisy actual pressure data and smooth theoretical tide data. ([#5749f4d](https://github.com/user/repo/commit/5749f4d725baa479be0e016116ecc80524d30a49))

### Refactor
- refactor: simplify astronomical features chart by using Plotly Express and a single y-axis. ([#5d2f7eb](https://github.com/user/repo/commit/5d2f7eb26ef8b2964f70a60a12dbcddc2698684a))

## [1.1.3] - 2026-02-22
### Features
- feat: Add lunar date display to the overview header and refactor the dual calendar metric using the `lunardate` library. ([#da30548](https://github.com/user/repo/commit/da30548bbe423bbce37205614ee4df6b883f78d3))
- feat: Integrate Skyfield for precise moon elevation calculations and enhance UI with lunar phase names and dual calendar dates. ([#09b3aac](https://github.com/user/repo/commit/09b3aacdeed29cad8ae074b7fa811dddaef40893))
- feat: Add a new plot for Residual Fluctuation centered at 0, showing positive and negative deviations. ([#cdfc373](https://github.com/user/repo/commit/cdfc373933e1abeffd4d03ac2291983d9e20d113))

### Refactor
- refactor: Enhance date and moon phase metric display with improved formatting, new labels, moon emojis, and illumination percentage. ([#1d7e640](https://github.com/user/repo/commit/1d7e6401c735e6f2d21ccbd6b006d116b83f4139))

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
