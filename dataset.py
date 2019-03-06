from __future__ import print_function
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, utils


def get_mnist_dataset(path):
    """Load dataset in local.

    Load train and test dataset in local.

    Args:
        path (str): Local path of dataset

    Returns:
        torchvision.datasets: Train dataset
        torchvision.datasets: Test dataset

    Example::
        get_dataset(./hoge)

    """
    train_dataset = datasets.MNIST(root=path, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))

    test_dataset = datasets.MNIST(root=path, train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

    return train_dataset, test_dataset


def load_dataset(path):
    """Load dataset in local.

    Load test image in local.
    Folder Tree

           Path
            ├─ Label 1
            |   ├─ Image 1
            |   ├─ Image 2
            |   └─ Image 3
            |
            └─ Label 2
                └─ Image 1

    Args:
        path (str): Directory path of dataset in Local

    Returns:
        List[(tensor.torch, int)]: dataset

    Example::
        get_dataset(./hoge)

    """
    datasets = []
    labels = os.listdir(path)
    for label in tqdm(labels, desc=" Label ", ascii=True):
        files = os.listdir(path + "/" + label)
        for file in tqdm(files, desc=" Data  ", ascii=True):
            img = Image.open(path + "/" + label + "/" + file).convert("L")
            torch_img = transforms.ToTensor()(img)
            data = (torch_img, int(label))
            datasets.append(data)
    print()
    return datasets


def save_mnist_dataset(dataset_path):
    def save(dataset, path):
        print(" Directory: {}".format(path))
        labels = {}
        for array, label in tqdm(dataset, desc=" Sequence ", ascii=True):
            index = labels.get(label) if labels.get(label) is not None else 0
            file = path + "/" + str(label) + "/" + "{}.png".format(index)
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            utils.save_image(array, file)
            labels[label] = index + 1

    train_dataset, test_dataset = get_mnist_dataset(dataset_path)
    train_path = dataset_path + "/MNIST" + "/train"
    test_path = dataset_path + "/MNIST" + "/test"
    save(train_dataset, train_path)
    save(test_dataset, test_path)
