import dataset
import os
import argparse
import os.path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=os.path.dirname(os.path.abspath(__file__)), help="Save path of images of datasets")
    args = parser.parse_args()

    print("\n Directory: {}\n".format(args.path))

    dataset.save_mnist_dataset(args.path)


if __name__ == '__main__':
    main()
