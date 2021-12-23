import h5py
import glob
import pandas as pd

files = glob.glob("data/**/*.parquet")

df = pd.read_parquet(files[0], columns=["review_body", "star_rating"])
df.to_csv("processed/data.csv", index=False)
print(df)
# dframe = pd.DataFrame(columns=["review_body", "star_rating"], dtype=[object, int])
