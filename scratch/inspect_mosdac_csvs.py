import zipfile
import io
import pandas as pd

z = zipfile.ZipFile('PS70-main.zip')

print("--- mosdac_needed_cyclones.csv ---")
df_needed = pd.read_csv(io.BytesIO(z.read('PS70-main/data/metadata/mosdac_needed_cyclones.csv')))
print(df_needed.info())
print(df_needed.head())

print("\n--- mosdac_priority1_named.csv ---")
df_p1 = pd.read_csv(io.BytesIO(z.read('PS70-main/data/metadata/mosdac_priority1_named.csv')))
print(df_p1.info())
print(df_p1.head(10))

print("\n--- mosdac_priority2_unnamed.csv ---")
df_p2 = pd.read_csv(io.BytesIO(z.read('PS70-main/data/metadata/mosdac_priority2_unnamed.csv')))
print(df_p2.info())
print(df_p2.head(10))
