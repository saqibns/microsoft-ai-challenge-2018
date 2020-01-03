import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os


def create_sample(data_file, seed, ones_frac, zeros_frac):
    print('Reading Data File')
    data = pd.read_csv(data_file, sep='\t', encoding='utf-8',
                       names=['qid', 'query', 'passage', 'label', 'pid'])
    ones = data[data['label'] == 1]
    zeros = data[data['label'] == 0]
    ones_down = ones.sample(frac=ones_frac, random_state=seed)
    zeros_down = zeros.sample(frac=zeros_frac, random_state=seed)
    df = pd.concat([ones_down, zeros_down], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print('Donwnsampled Data')
    return df


def save_sample(df, tsv_path, trn_save, val_save, test_size, seed):

    print('Saving files')
    if not os.path.exists(tsv_path):
        os.mkdir(tsv_path)

    trn, val = train_test_split(df, random_state=seed, test_size=test_size)
    if trn_save:
        train_path = os.path.join(tsv_path, 'train.tsv')
        trn.to_csv(train_path, sep='\t', encoding='utf-8', index=False, header=False)
    if val_save:
        val_path = os.path.join(tsv_path, 'dev.tsv')
        val.to_csv(val_path, sep='\t', encoding='utf-8', index=False, header=False)

    print('Train and Valid sets created')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data. Should contain the .tsv files for the task.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory to which the output files must be written.")

    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        required=True,
                        help="Random State to be used for sampling data.")

    parser.add_argument("--ones_frac",
                        default=None,
                        type=float,
                        required=True,
                        help="Random State to be used for sampling data.")

    parser.add_argument("--zeros_frac",
                        default=None,
                        type=float,
                        required=True,
                        help="Random State to be used for sampling data.")

    parser.add_argument("--train",
                        default=True,
                        action='store_true',
                        help="Whether training data required.")

    parser.add_argument("--dev",
                        default=False,
                        action='store_true',
                        help="Whether dev data required.")

    parser.add_argument("--dev_size",
                        default=0.0,
                        type=float,
                        help="Dev set size.")


    args = parser.parse_args()

    sample = create_sample(args.data_file, args.seed, args.ones_frac, args.zeros_frac)
    save_sample(sample, args.output_dir, args.train, args.dev, args.dev_size, args.seed)

if __name__ == '__main__':
    main()