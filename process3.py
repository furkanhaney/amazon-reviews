import glob
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    files = glob.glob("E:/Datasets/amazon-reviews/original/**/part-00000*.parquet")
    for file in files:
        print(file)
    print()

    df = pd.read_parquet(files, columns=["review_body", "star_rating"])
    df = df.rename(columns={"review_body": "text", "star_rating": "label",})
    df["label"] = df["label"] - 1
    print(df, "\n")
    df = df.sample(n=600 * 1000)
    train_df, test_df = train_test_split(df, test_size=20000, random_state=42)
    print(train_df, "\n")
    print(test_df, "\n")

    train_df.to_csv("processed/train.csv", index=False)
    test_df.to_csv("processed/test.csv", index=False)

    # text = "\n".join(train_df["review_body"].map(str).tolist())
    text = "\n".join(train_df["text"].map(str).tolist())
    print("Characters: {:,}".format(len(text)))
    with open("processed/raw0.txt", "w", encoding="utf-8") as file:
        file.write(text)
