import sys
import os

sys.path.append(os.getcwd())
from src.prepare_datasets import prepare_datasets


def main():
    # Test config
    train_dataset, validate_dataset, test_dataset = prepare_datasets()
    print(train_dataset[0].shape)


if __name__ == '__main__':
    main()
