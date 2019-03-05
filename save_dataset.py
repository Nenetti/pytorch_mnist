import dataset
import os
import argparse
import os.path


def main():
    current = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=current, help="Save path of images of datasets")
    args = parser.parse_args()

    print("\n Directory: {}\n".format(args.path))

    dataset.save_mnist_dataset(args.path)


if __name__ == '__main__':
    main()
