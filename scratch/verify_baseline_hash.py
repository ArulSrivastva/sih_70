import zipfile
import hashlib
import os

z = zipfile.ZipFile('PS70-main.zip')
baseline_bytes = z.read('PS70-main/models/detection/model_weights.pt')
baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()
print(f"Locked Baseline SHA256 in zip: {baseline_hash}")
print(f"Locked Baseline Size: {len(baseline_bytes)} bytes")

# Ensure models/detection directory exists on disk and extract if not present
os.makedirs('models/detection', exist_ok=True)
baseline_disk_path = 'models/detection/model_weights.pt'
if not os.path.exists(baseline_disk_path):
    with open(baseline_disk_path, 'wb') as f:
        f.write(baseline_bytes)
    print(f"Extracted baseline weights to {baseline_disk_path}")

with open(baseline_disk_path, 'rb') as f:
    disk_hash = hashlib.sha256(f.read()).hexdigest()

print(f"Disk Baseline SHA256: {disk_hash}")
print(f"Hashes match exactly: {baseline_hash == disk_hash}")
