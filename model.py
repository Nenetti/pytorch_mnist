import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch


class MnistModel(nn.Module):
    """
    MINIST model

    Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
    MaxPool2d(kernel_size=(2, 2))
    Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
    MaxPool2d(kernel_size=(2, 2))
    Linear(in_features=4 * 4 * 50, out_features=500)
    Linear(in_features=500, out_features=10)

    """

    '''
    Convolutional2D
    P = Padding, S = Stride, F = Filter
    W = Width, H = Height 

    
    W' = ((W + 2P -Fw) / S) + 1
    H' = ((H + 2P -Fh) / S) + 1
    '''

    optimizer = None
    train_loader = None
    test_loader = None
    device = None

    def __init__(self):
        super(MnistModel, self).__init__()
        # [1, 28, 28] -> [20, 24, 24]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        # [1, 24, 24] -> [20, 12, 12]
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        # [20, 12, 12] -> [50, 8, 8]
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        # [50, 8, 8] -> [50, 4, 4]
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        # [50, 4, 4] -> [800] -> [500]
        self.fc1 = nn.Linear(in_features=4 * 4 * 50, out_features=500)
        # [500] -> [10]
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        """Calculation forawrd.

        Override superclass function.

        Args:
            x (torch.tensor): input matrix

        Returns:
            torch.tensor: result of matrix calculation

        """
        # [batch, 1, 28, 28]
        x = F.relu(self.conv1(input=x))
        # [batch, 20, 24, 24]
        x = F.max_pool2d(input=x, kernel_size=(2, 2))
        # [batch, 20, 12, 12]
        x = F.relu(self.conv2(x))
        # [batch, 50, 8, 8]
        x = F.max_pool2d(input=x, kernel_size=(2, 2))
        # [batch, 50, 4, 4]
        x = x.reshape(-1, 4 * 4 * 50)
        # [batch, 800]
        x = F.relu(self.fc1(x))
        # [batch, 500]
        x = self.fc2(x)
        # [batch, 10]
        x = F.log_softmax(x, dim=1)
        return x

    def set_optimizer(self, optimizer):
        """Set optimizer.

        Set optimizer.

        Args:
            optimizer (torch.optim): optimizer

        Example::
            torch.optim.adam.Adam()
        """
        self.optimizer = optimizer

    def set_test_loader(self, test_loader):
        """Set test_loader.

        Set test_loader.

        Args:
            test_loader (torch.utils.data.dataloader.DataLoader): test_loader
        """
        self.test_loader = test_loader

    def set_train_loader(self, train_loader):
        """Set train_loader.

        Set train_loader.

        Args:
            train_loader (torch.utils.data.dataloader.DataLoader): train_loader
        """
        self.train_loader = train_loader

    def set_device(self, device):
        """Set device.

        Set device type cpu or cuda.

        Args:
            device (torch.device): device
        """
        self.device = device
        self.to(device)

    def run(self, epoch):
        """run train and test.

        Run train and test for 'epoch' times.

        Args:
            epoch (int): Number of epoch
        """
        for _ in tqdm(range(epoch), desc="       Epoch ", ascii=True):
            self.run_training()
            self.run_test()

    def run_training(self):
        """run train.

        Run training to calculate one epoch.

        """
        self.train()
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=" Train_Batch ", ascii=True)):
            # type: (int, (torch.tensor, torch.tensor))
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            # if batch_idx % 10 == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, batch_idx * len(data), len(train_loader.dataset),
            #               100. * batch_idx / len(train_loader), loss.item()))
        print()

    def run_test(self):
        """run test.

        Run test to calculate one epoch.
        Calculate average loss and accuracy

        """
        self.eval()     # == self.train(False)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="  Test_Batch ", ascii=True):
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        print()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def predict(self, data):
        """Calculation input data.

        Use trained model to calculate input data

        Args:
            data (torch.tensor): input matrix

        Returns:
            torch.tensor: the max score in output

        """
        self.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self(data)
            return output
