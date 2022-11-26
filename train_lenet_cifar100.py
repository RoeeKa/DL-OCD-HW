import Lenet5

from torchvision import transforms
import torch.nn as nn
import torchvision
import torch

import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_test(testloader):
    total_items = 0
    correct_items = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            correct_items += predicted.eq(labels).sum().item()
            total_items += labels.size(0)
    return correct_items / total_items

transform = transforms.Compose([
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

model = Lenet5.NetOriginal(ch_in=3)
model = model.to(device)


def train(epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0
    for epoch_num in range(epochs):
        running_loss = 0
        total_items = 0
        correct_items = 0
        for idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            correct_items += predicted.eq(labels).sum().item()
            total_items += labels.size(0)
            running_loss += loss.item()

        test_acc = eval_test(testloader)
        print(f'Finished epoch: {epoch_num + 1}, loss: {running_loss / len(trainset)}, acc: {correct_items / total_items}, test_acc: {test_acc}')

        if best_acc < test_acc:
            best_acc = test_acc
            state = {
                'net': model.state_dict(),
                'acc': best_acc,
                'epoch': epoch_num + 1,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, './checkpoints/lenetcifar100' + '.pth.tar')


train(10000)

