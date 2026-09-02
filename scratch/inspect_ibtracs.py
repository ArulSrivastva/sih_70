import zipfile
import io
import pandas as pd

z = zipfile.ZipFile('PS70-main.zip')
df_ib = pd.read_csv(io.BytesIO(z.read('PS70-main/data/metadata/ibtracs_clean.csv')))
print("ibtracs_clean.csv shape:", df_ib.shape)
print("Columns:", df_ib.columns.tolist())
print(df_ib.head(5))
print("\nUnique cyclones in IBTrACS:", df_ib['cyclone_id'].nunique())
print("Date range in IBTrACS:", df_ib['iso_time'].min(), "to", df_ib['iso_time'].max())
