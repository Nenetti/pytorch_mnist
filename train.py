from __future__ import print_function
import torch
import torch.optim as optim
import os
from model import MnistModel
import torch.utils.data
import argparse
import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default="trained_model.pt", help="Load path of Trained model in local")
    parser.add_argument('--save_model', default="trained_model.pt", help='Save path of Trained model in local')
    parser.add_argument('--batch', default=100, help='Save path of Trained model in local')
    parser.add_argument('--epoch', default=10, help='Save path of Trained model in local')
    parser.add_argument('--lr', default=0.01, help='Save path of Trained model in local')
    parser.add_argument('--device', default="cuda", help='Save path of Trained model in local')
    parser.add_argument('--dataset', default=os.path.dirname(os.path.abspath(__file__)), help='Load path of image dataset in local')

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

    #train_dataset, test_dataset = dataset.get_mnist_dataset(args.dataset)
    train_dataset = dataset.load_dataset(args.dataset+"/MNIST/train")
    test_dataset = dataset.load_dataset(args.dataset+"/MNIST/test")

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
