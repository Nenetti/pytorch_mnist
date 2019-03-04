from __future__ import print_function
import torch
from torchvision import datasets, transforms
import os
from model import MnistModel
import torch.utils.data
import argparse
from tqdm import tqdm
from PIL import Image


def get_dataset(path):
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
    parser.add_argument('--device', default="cuda", help='Save path of Trained model in local')
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    load_model = args.load_model

    print("Device: {}:".format(device))
    print("Load Model: {}:".format(load_model))
    print()

    model = MnistModel()

    if os.path.exists(load_model):
        model.load_state_dict(torch.load(load_model, device))
    else:
        print("Warning: Load Model '{}' is not exist".format(load_model))
        exit(1)

    model.set_device(device)

    test_dataset = load_dataset(os.environ["HOME"] + "/dataset/MNIST/test")
    # hoge, test_dataset = get_dataset(os.environ["HOME"] + "/dataset")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)
    model.set_test_loader(test_loader)
    model.run_test()

    '''
    correct = 0
    accuracy = tqdm(total=len(test_dataset), desc="Accuracy", ascii=True)
    for i, data in enumerate(tqdm(test_dataset, desc="Test", ascii=True)):
        image, label = data
        # to convert 1 batch size
        result = model.predict(image.reshape(1, 1, 28, 28))
        result = result.argmax(dim=1, keepdim=True).reshape(-1).cpu().numpy()[0]
        if not int(label) == result:
            pass
            # print("Correct = {}".format(label), " -> ", "Predict = {}".format(result))
            # plt.imshow(image.reshape(28, 28), cmap="gray")
            # plt.show()
        else:
            correct = correct+1
            accuracy.update()
    #print("Result: {}/{}".format(correct, len(test_dataset)))
    '''

if __name__ == '__main__':
    main()
