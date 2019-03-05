from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from model import MnistModel
import torch.utils.data
import argparse
from tqdm import tqdm
from PIL import Image

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
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    test_dataset = datasets.MNIST(root=path, train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default="trained_model.pt", help="Load path of Trained model in local")
    parser.add_argument('--save_model', default="trained_model.pt", help='Save path of Trained model in local')
    parser.add_argument('--batch', default=100, help='Save path of Trained model in local')
    parser.add_argument('--epoch', default=10, help='Save path of Trained model in local')
    parser.add_argument('--lr', default=0.01, help='Save path of Trained model in local')
    parser.add_argument('--device', default="cuda", help='Save path of Trained model in local')
    args = parser.parse_args()

    batch_size = args.batch
    epoch = args.epoch
    lr = args.lr
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    load_model = args.load_model
    save_model = args.save_model

    print("Batch_size: {}:".format(batch_size))
    print("Epoch: {}:".format(epoch))
    print("Learning rate: {}:".format(lr))
    print("Device: {}:".format(device))
    print("Load Model: {}:".format(load_model))
    print("Save Model: {}:".format(save_model))
    print()

    train_dataset, test_dataset = get_mnist_dataset(os.environ["HOME"] + "/dataset")
    #train_dataset = load_dataset(os.environ["HOME"] + "/dataset/MNIST/train")
    #test_dataset = load_dataset(os.environ["HOME"] + "/dataset/MNIST/test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = MnistModel()

    if os.path.exists(load_model):
        model.load_state_dict(torch.load(load_model, device))
    else:
        print("Warning: Load Model '{}' is not exist".format(load_model))

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.set_test_loader(test_loader)
    model.set_train_loader(train_loader)
    model.set_optimizer(optimizer)
    model.set_device(device)

    model.run(epoch)

    torch.save(model.state_dict(), save_model)


if __name__ == '__main__':
    main()
