import os
import glob
import zipfile

# Search in filesystem
matches = glob.glob('**/model_weights.pt', recursive=True) + glob.glob('**/*detector*.pt', recursive=True)
print("Filesystem matches:")
for m in matches:
    print(" ", m)

# Search in zip
z = zipfile.ZipFile('PS70-main.zip')
zip_matches = [f for f in z.namelist() if 'model_weights' in f or 'detection' in f and f.endswith('.pt')]
print("\nZip matches:")
for zm in zip_matches:
    print(" ", zm)
