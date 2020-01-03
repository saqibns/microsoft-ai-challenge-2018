import pandas as pd
import argparse


def adjust_and_save(eval_file_path):
    data = pd.read_csv(eval_file_path, sep='\t', encoding='utf-8',
                       names=['qid', 'query', 'passage', 'pid'])
    data['label'] = 0
    data = data[['qid', 'query', 'passage', 'label', 'pid']]
    data.to_csv(eval_file_path, sep='\t', encoding='utf-8', index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="Eval TSV to be modified.")

    args = parser.parse_args()

    adjust_and_save(args.data_file)


if __name__ == '__main__':
    main()

