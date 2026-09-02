import zipfile
import io

z = zipfile.ZipFile('PS70-main.zip')
insat_dirs = set()
for f in z.namelist():
    if 'insat' in f.lower():
        parts = f.split('/')
        if len(parts) > 3:
            insat_dirs.add('/'.join(parts[:4]))

print("Subdirectories in raw insat:")
for d in sorted(insat_dirs):
    print(" ", d)

# Count files per subdirectory
dir_counts = {}
for f in z.namelist():
    if 'insat' in f.lower() and not f.endswith('/'):
        d = '/'.join(f.split('/')[:4])
        dir_counts[d] = dir_counts.get(d, 0) + 1

print("\nFile counts per subdirectory:")
for d, count in sorted(dir_counts.items()):
    print(f"  {d}: {count} files")
