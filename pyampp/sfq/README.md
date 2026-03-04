# SFQ In pyAMPP

This folder contains the Python rewrite of SFQ (Super Fast and Quality) magnetic-field disambiguation routines used by AMPP/GX workflows.

## Source And Attribution

The SFQ method and reference implementation come from the IDL package maintained by Sergey Anfinogentov.

- Original source repository: https://github.com/Sergey-Anfinogentov
- Authors: George Rudenko, Sergey Anfinogentov

Please cite the original SFQ work and repository when using these routines in scientific outputs.

## Integration Note

The current implementation is incremental and parity-focused. Some low-level geometry/FFT backends are still injected as callables while their native Python implementations are finalized.
