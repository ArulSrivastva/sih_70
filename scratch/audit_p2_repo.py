import os
import glob
import json
import zipfile
import io
import pandas as pd

# Check files in repository
z = zipfile.ZipFile('PS70-main.zip')
namelist = z.namelist()

detection_files = [f for f in namelist if 'detection' in f.lower() or 'image' in f.lower() or 'mosdac' in f.lower()]
print("Detection & Image files in zip:")
for f in detection_files[:30]:
    print(" ", f)

# Check for track data in IBTrACS / P1 / P4
track_files = [f for f in namelist if 'ibtracs' in f.lower() or 'track' in f.lower() or 'cyclone' in f.lower()]
print("\nTrack & Cyclone metadata files in zip:")
for f in track_files[:30]:
    print(" ", f)

# Check workspace filesystem for any existing MOSDAC or crop scripts
ws_files = glob.glob("**/*mosdac*", recursive=True) + glob.glob("**/*crop*", recursive=True) + glob.glob("**/*detection*", recursive=True)
print("\nWorkspace matches for mosdac / crop / detection:")
for f in ws_files:
    print(" ", f)
