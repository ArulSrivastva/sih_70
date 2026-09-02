import zipfile
import io

z = zipfile.ZipFile('PS70-main.zip')
src_files = [f for f in z.namelist() if f.startswith('PS70-main/src/')]
for f in sorted(src_files):
    print(f)
