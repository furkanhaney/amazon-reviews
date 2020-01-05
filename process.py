import re
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from nltk import word_tokenize


def get_data():
    dframe = pd.DataFrame(columns=["review_body", "star_rating"], dtype=[object, int])
    path = "C:/machine_learning/datasets_original/amazon_reviews/parquet/"


def clean_review(text):
    text = text.lower()
    for code in [r"\\", "<br />", "<BR>"]:
        text = text.replace(code, "")
    text = text.replace("&quot;", "'")
    text = re.sub(r"\[\[.*]]", "", text)
    return word_tokenize(text)


def print_summary(df):
    print("Average Length: {:.0f}".format(df["review_body"].str.len().mean()))
    for i in range(6, 12):
        num = 2 ** i
        ratio = len(df[df["review_body"].str.len() >= num]) / len(df)
        print("{:.2%} of reviews longer than {} words.".format(ratio, num))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max_words", type=int, default=512)
    args = parser.parse_args()

    pool = mp.Pool(args.threads)
    df = pd.read_parquet(args.path, columns=["review_body", "star_rating"])
    df = df[df["review_body"].str.len() >= 10]
    reviews = list(df["review_body"])
    ratings = list(df["star_rating"])

    word_set = {}
    reviews_array = np.zeros((len(reviews), args.max_words), dtype=np.int64)
    ratings = np.array(ratings, dtype=np.int64)

    pbar = tqdm(desc="Processing the text.", total=len(reviews), ascii=True)
    for i, review in enumerate(pool.imap(clean_review, reviews)):
        for j, word in enumerate(review):
            if j == args.max_words:
                break
            if word not in word_set:
                word_set[word] = len(word_set) + 1
            reviews_array[i, j] = word_set[word]
        pbar.update()
    pbar.close()

    print("Dictionary Size: {:,}".format(len(word_set)))
    np.save("data/reviews.npy", reviews_array)
    np.save("data/ratings.npy", ratings)


if __name__ == "__main__":
    main()
