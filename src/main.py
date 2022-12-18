from src.prepare_datasets import prepare_datasets


def main():
    # Test config
    train_dataset, validate_dataset, test_dataset = prepare_datasets()


if __name__ == '__main__':
    main()
