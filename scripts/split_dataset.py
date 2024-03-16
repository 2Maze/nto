import argparse
import pandas as pd
import os

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        type=str,
        help='Path to dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./data',
        help='Path to output folder'
    )
    parser.add_argument(
        '--test',
        type=float,
        help='Test part'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = pd.read_csv(args.input)
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=args.test,
        random_state=args.seed,
        stratify=dataset['name']
    )
    dataset.iloc[train_indices].to_csv(os.path.join(args.output, 'train.csv'))
    dataset.iloc[test_indices].to_csv(os.path.join(args.output, 'val.csv'))


if __name__ == '__main__':
    main()
