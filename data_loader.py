import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from torchvision.datasets import mnist
from torchvision import transforms
import Lenet5
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from copy import deepcopy

from muxnet.muxnet import muxnet_m
from hcgnet.HCGNet_CIFAR import HCGNet_A1


def _data_transforms(args):
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
        transforms.RandomHorizontalFlip(),
    ])

    # if args.autoaugment:
    #     train_transform.transforms.append(CIFAR10Policy())

    train_transform.transforms.append(transforms.ToTensor())

    # if args.cutout:
    #     train_transform.transforms.append(Cutout(args.cutout_length))

    train_transform.transforms.append(transforms.Normalize(norm_mean, norm_std))

    valid_transform = transforms.Compose([
        transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    return train_transform, valid_transform


def wrapper_dataset(config, args, device):
    if args.datatype == 'tinynerf':
        
        data =  np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] =get_minibatches
        batch['chunksize'] =chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [],[]
        for img,tfrom in zip(images,tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose(
                            [
                            transforms.ToTensor()
                            ])
        train_dataset = mnist.MNIST(
                "\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST(
                "\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [],[]
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            test_ds.append(deepcopy(batch))
    elif args.datatype == 'muxnet-cifar100':
        class Cutout(object):
            def __init__(self, length):
                self.length = length

            def __call__(self, img):
                h, w = img.size(1), img.size(2)
                mask = np.ones((h, w), np.float32)
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.
                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img *= mask
                return img

        train_ds, test_ds = [],[]
        train_transform, valid_transform = _data_transforms(args)
        train_data = torchvision.datasets.CIFAR100(
            root='data', train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(
            root='data', train=False, download=True, transform=valid_transform)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = F.interpolate(inputs, size=224, mode='bicubic', align_corners=False)
            batch = {'input':inputs,'output':targets}
            train_ds.append(deepcopy(batch))
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs = F.interpolate(inputs, size=224, mode='bicubic', align_corners=False)
            batch = {'input':inputs,'output':targets}
            test_ds.append(deepcopy(batch))
        model = muxnet_m(pretrained=False, num_classes=100)
    elif args.datatype == 'hcgnet':
        train_ds, test_ds = [], []
        trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.RandomCrop(32, padding=4),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                          [0.2675, 0.2565, 0.2761])
                                                 ]))

        testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                         [0.2675, 0.2565, 0.2761]),
                                                ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,
                                                  pin_memory=(torch.cuda.is_available()))

        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                                 pin_memory=(torch.cuda.is_available()))

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch = {'input':inputs,'output':targets}
            train_ds.append(deepcopy(batch))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch = {'input':inputs,'output':targets}
            test_ds.append(deepcopy(batch))

        model = HCGNet_A1(num_classes=100)
    elif args.datatype == 'lenet-cifar':
        train_ds, test_ds = [], []

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 1

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch = {'input':inputs,'output':targets}
            train_ds.append(deepcopy(batch))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch = {'input':inputs,'output':targets}
            test_ds.append(deepcopy(batch))

        model = Lenet5.NetOriginal(ch_in=3)
    elif args.datatype == 'muxnet-cifar10':
        train_transform, valid_transform = _data_transforms(args)
        train_ds, test_ds = [],[]

        train_data = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True, transform=valid_transform)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = F.interpolate(inputs, size=224, mode='bicubic', align_corners=False)
            batch = {'input':inputs,'output':targets}
            train_ds.append(deepcopy(batch))

            if idx == 100:
                break
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs = F.interpolate(inputs, size=224, mode='bicubic', align_corners=False)
            batch = {'input':inputs,'output':targets}
            test_ds.append(deepcopy(batch))

            if idx == 100:
                break
        model = muxnet_m(pretrained=True, num_classes=10)
    else:
        raise Exception('Bla')
    return train_ds,test_ds,model
