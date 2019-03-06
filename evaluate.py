from __future__ import print_function
import torch
import os
from model import MnistModel
import torch.utils.data
import argparse
import dataset
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default="trained_model.pt", help="Load path of Trained model in local")
    parser.add_argument('--device', default="cuda", help='Save path of Trained model in local')
    parser.add_argument('--file', default=os.path.dirname(os.path.abspath(__file__)), help='Load path of image dataset in local')
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

    img = Image.open(args.file).convert("L")
    data = torch.stack([transforms.ToTensor()(img)])

    result = model.predict(data)
    result = result.argmax(dim=1, keepdim=True).reshape(-1).cpu().numpy()[0]

    print("Result: {}".format(result))

    plt.title(result)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
