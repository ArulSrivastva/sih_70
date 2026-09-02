# Available Environmental Data -- P4 Phase 2

## Raw Features

| Feature | Source | Temporal | Spatial | Safe |
|---------|--------|----------|---------|------|
| lat | IBTrACS | 3-hourly | storm center | True |
| lon | IBTrACS | 3-hourly | storm center | True |
| wind_speed | IBTrACS cross-agency | 3-hourly | storm center | True |
| pressure | IBTrACS cross-agency | 3-hourly | storm center | True |
| sst | ERA5 single-level | 3-hourly | storm center | True |
| wind_u | ERA5 10m | 3-hourly | storm center | True |
| wind_v | ERA5 10m | 3-hourly | storm center | True |

## Derived Features in Feature Dataset

| Feature | Formula | Safe |
|---------|---------|------|
| delta_lat | lat(t) - lat(t-1) | Yes |
| delta_lon | wrapped lon(t) - lon(t-1) | Yes |
| movement_speed | haversine(prev,cur) / 6h | Yes |
| movement_direction | bearing prev->cur | Yes |
| wind_change | wind(t) - wind(t-1) | Yes |
| pressure_change | pres(t) - pres(t-1) | Yes |
| sst_change | sst(t) - sst(t-1) | Yes |
| environmental_wind_speed | sqrt(u^2+v^2) | Yes |
| environmental_wind_direction | atan2(-u,-v) | Yes |

## Unavailable Variables

- pressure_levels (500hPa, 850hPa, 200hPa) -- not downloaded
- vertical_wind_shear -- requires multi-level data
- relative_humidity -- not downloaded
- geopotential_height -- not downloaded
- vorticity -- not downloaded
- divergence -- not downloaded
- SST_anomaly -- not computed

## Key Notes

- Only single-level ERA5 reanalysis available
- No multi-level atmospheric data in the codebase
- Environmental wind features show negligible predictive power in Phase-1 ablation
- SST has ~28% NaN in P3 classification data
