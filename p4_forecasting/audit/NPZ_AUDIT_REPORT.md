# NPZ AUDIT REPORT (P4 PHASE 1)


Total sequences: 3076 (train+val+test = 2275+378+423 = 3076).

### Split: train

Keys: ['X', 'Y', 'features', 'targets']

- `X` shape=(2275, 5, 7) dtype=float32 NaN=0 +inf/-inf=0 min=-25.429855 max=1006.000000 mean=170.438309 std=337.513519
- `Y` shape=(2275, 3, 3) dtype=float32 NaN=0 +inf/-inf=0 min=3.600000 max=240.800003 mean=57.970188 std=40.850800

features array: [np.str_('lat'), np.str_('lon'), np.str_('wind_speed'), np.str_('pressure'), np.str_('sst'), np.str_('wind_u'), np.str_('wind_v')]
targets array : [np.str_('lat'), np.str_('lon'), np.str_('wind_speed')]

### Split: val

Keys: ['X', 'Y', 'features', 'targets']

- `X` shape=(378, 5, 7) dtype=float32 NaN=0 +inf/-inf=0 min=-16.851105 max=1008.000000 mean=170.081375 std=339.435028
- `Y` shape=(378, 3, 3) dtype=float32 NaN=0 +inf/-inf=0 min=7.900000 max=148.199997 mean=56.210098 std=33.252312

features array: [np.str_('lat'), np.str_('lon'), np.str_('wind_speed'), np.str_('pressure'), np.str_('sst'), np.str_('wind_u'), np.str_('wind_v')]
targets array : [np.str_('lat'), np.str_('lon'), np.str_('wind_speed')]

### Split: test

Keys: ['X', 'Y', 'features', 'targets']

- `X` shape=(423, 5, 7) dtype=float32 NaN=0 +inf/-inf=0 min=-19.245972 max=1005.000000 mean=170.167374 std=336.948181
- `Y` shape=(423, 3, 3) dtype=float32 NaN=0 +inf/-inf=0 min=5.900000 max=185.199997 mean=58.332439 std=39.264290

features array: [np.str_('lat'), np.str_('lon'), np.str_('wind_speed'), np.str_('pressure'), np.str_('sst'), np.str_('wind_u'), np.str_('wind_v')]
targets array : [np.str_('lat'), np.str_('lon'), np.str_('wind_speed')]
