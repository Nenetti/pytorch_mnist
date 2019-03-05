from __future__ import print_function
import torch
import os
from model import MnistModel
import torch.utils.data
import argparse
import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default="trained_model.pt", help="Load path of Trained model in local")
    parser.add_argument('--device', default="cuda", help='Save path of Trained model in local')
    parser.add_argument('--dataset', default=os.path.dirname(os.path.abspath(__file__)), help='Load path of image dataset in local')
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

    test_dataset = dataset.load_dataset(args.dataset+"/MNIST/test")
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
