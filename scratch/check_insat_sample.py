import zipfile
import io

z = zipfile.ZipFile('PS70-main.zip')
insat_raw_files = [f for f in z.namelist() if f.startswith('PS70-main/data/raw/insat/') and not f.endswith('/')]
print(f"Total files in data/raw/insat/: {len(insat_raw_files)}")
print("Sample files:")
for f in sorted(insat_raw_files)[:20]:
    print(" ", f)
