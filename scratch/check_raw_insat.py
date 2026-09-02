import zipfile
import io

z = zipfile.ZipFile('PS70-main.zip')
raw_insat = [f for f in z.namelist() if 'insat' in f.lower()]
print(f"Total insat files in zip: {len(raw_insat)}")
for f in raw_insat[:25]:
    print(" ", f)
