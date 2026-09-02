# PHYSICAL SANITY REPORT (P4 PHASE 1)

Checks performed on X (input) and Y (target) values before any scaling. Nominal ranges used: |lat|<=90, |lon|<=360, wind>=0, 850<=pressure<=1080 hPa, 0<=sst<=45 (deg C, as delivered), |u|,|v|<=100 m/s.

## Split: train (violations: 0)

- max |lat| (X)      : 26.0000
- max |lon| (X)      : 141.0000
- wind km/h (X)      : 37.0000 .. 240.8000
- pressure hPa (X)   : 920.00 .. 1006.00
- sst deg C (X)      : 25.4900 .. 31.4100
- max |u|,|v| m/s (X): 25.4299 / 23.3893
- max |lat| (Y)      : 26.3000
- max |lon| (Y)      : 135.1000
- max wind km/h (Y)  : 240.8000

## Split: val (violations: 0)

- max |lat| (X)      : 24.4000
- max |lon| (X)      : 118.2000
- wind km/h (X)      : 37.0000 .. 148.2000
- pressure hPa (X)   : 972.00 .. 1008.00
- sst deg C (X)      : 27.2800 .. 30.8400
- max |u|,|v| m/s (X): 14.6048 / 16.8511
- max |lat| (Y)      : 24.4000
- max |lon| (Y)      : 112.8000
- max wind km/h (Y)  : 148.2000

## Split: test (violations: 0)

- max |lat| (X)      : 24.4000
- max |lon| (X)      : 93.3000
- wind km/h (X)      : 37.0000 .. 185.2000
- pressure hPa (X)   : 950.00 .. 1005.00
- sst deg C (X)      : 25.6100 .. 31.6600
- max |u|,|v| m/s (X): 18.3499 / 19.2460
- max |lat| (Y)      : 26.0000
- max |lon| (Y)      : 91.8000
- max wind km/h (Y)  : 185.2000

Important unit note: `sst` in the npz is DEGREES CELSIUS (range ~25..31), matching P1's `sst_celsius` conversion. ERA5 raw registers SST in Kelvin. The Phase-1 spec/summary assumed Kelvin; the data is Celsius. No values were altered (follows 'do not silently alter values' rule). The Kelvin conversion is a documented Phase-2 decision.
