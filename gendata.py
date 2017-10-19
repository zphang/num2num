import argparse
import collections as col
import csv
import numpy as np
import pandas as pd

import num2words
import tqdm


def generate_dataset(size, low, high, seed=None):
    if seed is not None:
        np.random.seed(seed)
    num_data = np.random.randint(
        low=low,
        high=high,
        size=size,
    )
    df_dict = col.OrderedDict([
        ("word", []),
        ("num", []),
    ])
    for i in tqdm.tqdm(num_data, mininterval=1.):
        df_dict["word"].append(num2words.num2words(i))
        df_dict["num"].append(i)
    return pd.DataFrame(df_dict)


def save_dataset(df, filepath):
    df.to_csv(filepath, index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)


def main():
    parser = argparse.ArgumentParser("Generate random num2num dataset csv",
                                     add_help=False)
    parser.add_argument("-o", "--output_path", help="Output path",
                        required=True)
    parser.add_argument("-s", "--size", help="Number of samples",
                        required=True)
    parser.add_argument("--low", help="Lowest number (inclusive)",
                        required=False, default=-10000000)
    parser.add_argument("--high", help="Highest number (exclusive)",
                        required=False, default=10000000)
    parser.add_argument("--seed", help="NumPy Seed",
                        required=False, default=0)
    args = parser.parse_args()

    dataset = generate_dataset(
        size=int(args.size),
        low=int(args.low),
        high=int(args.high),
        seed=int(args.seed),
    )
    save_dataset(
        df=dataset,
        filepath=args.output_path,
    )


if __name__ == "__main__":
    main()
