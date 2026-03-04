# IDL .sav ↔ HDF5 mapping (b3d_data_20240512T000000_dim64x64x64)

Files compared
- IDL .sav: `/Users/gelu/gx_models/2024-05-12/hmi.M_720s.20240511_235843.E73S3CR.CEA.NAS.CHR.sav`
- HDF5: `/Users/gelu/pyampp/gx_models/b3d_data_20240512T000000_dim64x64x64.h5`

## Field-by-field mapping

| IDL BOX field | HDF5 dataset | IDL shape | HDF5 shape | Notes |
| --- | --- | --- | --- | --- |
| `DR` | `chromo/dr` | (3,) | (3,) |  |
| `STARTIDX` | `chromo/start_idx` | (64,64,64) | (262144,) | flattened in HDF5 |
| `ENDIDX` | `chromo/end_idx` | (64,64,64) | (262144,) | flattened in HDF5 |
| `AVFIELD` | `chromo/av_field` | (64,64,64) | (262144,) | flattened in HDF5 |
| `PHYSLENGTH` | `chromo/phys_length` | (64,64,64) | (262144,) | flattened in HDF5 |
| `BCUBE` | `chromo/bcube` | (3,64,64,64) | (64,64,64,3) | axis order differs |
| `CHROMO_IDX` | `chromo/chromo_idx` | (328170,) | (328090,) | length differs |
| `CHROMO_BCUBE` | `chromo/chromo_bcube` | (3,90,64,64) | (64,64,90,3) | axis order differs |
| `N_HTOT` | `chromo/n_htot` | (328170,) | (328090,) | length differs |
| `N_HI` | `chromo/n_hi` | (328170,) | (328090,) | length differs |
| `N_P` | `chromo/n_p` | (328170,) | (328090,) | length differs |
| `DZ` | `chromo/dz` | (152,64,64) | (64,64,152) | axis order differs |
| `CHROMO_N` | `chromo/chromo_n` | (328170,) | (328090,) | length differs |
| `CHROMO_T` | `chromo/chromo_t` | (328170,) | (328090,) | length differs |
| `CHROMO_LAYERS` | `chromo/chromo_layers` | () | () |  |
| `TR` | `chromo/tr` | (64,64) | (64,64) |  |
| `TR_H` | `chromo/tr_h` | (64,64) | (64,64) |  |
| `CORONA_BASE` | `chromo/corona_base` | () | () |  |

## HDF5-only datasets (not present in IDL BOX)

- `chromo/apex_idx`
- `chromo/codes`
- `chromo/seed_idx`
- `chromo/voxel_status`
- `chromo/chromo_mask` (IDL equivalent appears as `BOX.BASE.CHROMO_MASK`)
- `corona/bx`, `corona/by`, `corona/bz` (3D field cubes, `model_type` attr indicates POT/NLFFF)

## IDL-only BOX fields (not present in HDF5)

- `ADD_BASE_LAYER`
- `STATUS` (64,64,64 uint8)
- `ID`
- `EXECUTE`
- `INDEX` (WCS/observation metadata)
- `REFMAPS` (IDL object metadata)
- `BASE` (struct with 2D fields)
  - `BX`, `BY`, `BZ` (64,64)
  - `IC` (64,64)
  - `CHROMO_MASK` (64,64)

## Metadata differences

- HDF5 `chromo` group attributes: `dsun_obs`, `lat`, `lon`, `obs_time`
- IDL `INDEX` includes WCS, pointing, and history/comment fields not present as HDF5 attrs.

## Regeneration

Use the script in `pyampp/tests/compare_idl_hdf5.py` to generate a JSON report:

```bash
python pyampp/tests/compare_idl_hdf5.py \
  --h5 /Users/gelu/pyampp/gx_models/b3d_data_20240512T000000_dim64x64x64.h5 \
  --sav /Users/gelu/gx_models/2024-05-12/hmi.M_720s.20240511_235843.E73S3CR.CEA.NAS.CHR.sav \
  --out /tmp/idl_hdf5_report.json
```
