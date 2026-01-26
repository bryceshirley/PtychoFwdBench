# SSSP

This code implements the Spectral Split-Step Padé method in python. It generates the figures from the corresponding (yet unpublished) article ["A Spectral Split-Step Padé Method for Guided Wave Propagation"](https://www.imacm.uni-wuppertal.de/fileadmin/imacm/preprints/2025/imacm_25_11.pdf)

## Structure

Contains the core methods as well as the plot generation at the bottom of the file.

### modes

Contains normal mode starters, both summed up to form initial conditions as well as the raw modes. for both 75 Hz and 100 Hz.

### pkg

Routines for the Padé approximation, computation of Padé coefficients from Taylor coefficients as well as a routine to generate the Taylor coefficients for the exponential of a square root.
