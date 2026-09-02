import zipfile
import io

z = zipfile.ZipFile('PS70-main.zip')

print("=== preprocess_satellite.py ===")
print(z.read('PS70-main/src/data/preprocess_satellite.py').decode('utf-8'))

print("\n=== detection_dataset.py ===")
print(z.read('PS70-main/src/data/detection_dataset.py').decode('utf-8'))
