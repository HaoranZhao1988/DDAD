from torchvision import datasets, transforms
import torch
import os
from dataset.cityscapes1 import Cityscapes
from dataset.caltech import Caltech101
from dataset.camvid import CamVid
from dataset.voc import VOCSegmentation
from dataset.nyu import NYUv2, NYUv2Depth
from utils import ext_transforms



def get_dataloader(args):
    if args.dataset.lower()=='mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    elif args.dataset.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower()=='cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_root, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower()=='caltech101':
        train_loader = torch.utils.data.DataLoader(
            Caltech101(args.data_root, train=True, download=args.download,
                        transform=transforms.Compose([
                            transforms.Resize(128),
                            transforms.RandomCrop(128),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            Caltech101(args.data_root, train=False, download=args.download,
                        transform=transforms.Compose([
                            transforms.Resize(128),
                            transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    elif args.dataset.lower()=='imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_val_dataset_dir = os.path.join(args.data_root, "train")
        test_dataset_dir = os.path.join(args.data_root, "val")

        trainset = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train)
        valset = datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=5,
                                                  pin_memory=True)
        test_loader = torch.utils.data.DataLoader(valset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=5,
                                                pin_memory=True)

    ########### Segmentation
    elif args.dataset.lower()=='camvid':
        print(args.data_root)
        train_loader = torch.utils.data.DataLoader(
            CamVid(args.data_root, split='train',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                            ext_transforms.ExtRandomHorizontalFlip(),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            CamVid(args.data_root, split='test',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    elif args.dataset.lower() in ['nyuv2']:
        train_loader = torch.utils.data.DataLoader(
            NYUv2(args.data_root, split='train',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                            ext_transforms.ExtRandomHorizontalFlip(),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            NYUv2(args.data_root, split='test',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    elif args.dataset.lower() in ['cityscapes']:
        train_loader = torch.utils.data.DataLoader(
            Cityscapes(args.data_root, split='train',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                            ext_transforms.ExtRandomHorizontalFlip(),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            Cityscapes(args.data_root, split='val',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    elif args.dataset.lower() in ['voc2012']:
        train_loader = torch.utils.data.DataLoader(
            VOCSegmentation(args.data_root, image_set='train',
                        transform=ext_transforms.ExtCompose([
                            # ext_transforms.ExtRandomScale((0.5, 2.0)),
                            # ext_transforms.ExtRandomCrop(513, pad_if_needed=True),
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                            ext_transforms.ExtRandomHorizontalFlip(),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            VOCSegmentation(args.data_root, image_set='val',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtCenterCrop(224),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

