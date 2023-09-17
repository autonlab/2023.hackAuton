import torchvision
import torchvision.transforms as transforms



def load_dataset(data, size, transform_train, transform_test, data_dir=None):
    if data_dir is None:
        data_dir = "../" + data

    if data == "coco":
        trainset = torchvision.datasets.CocoDetection(root=data_dir,transform=transform_train)
        testset = torchvision.datasets.CocoDetection(root=data_dir,transform=transform_test)
    elif data == "medical":
        trainset = torchvision.datasets.ImageFolder(root="/ocean/projects/cis230067p/cteh/cxr14/train",transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root="/ocean/projects/cis230067p/cteh/cxr14/test",transform=transform_test)
    elif data == "braintumor":
        trainset = torchvision.datasets.ImageFolder(root="/jet/home/guntakan/braintumor/Training",transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root="/jet/home/guntakan/braintumor/Testing",transform=transform_test)
    elif data == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        # testloader = torch.utils.dta.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "flower":
        trainset = torchvision.datasets.Flowers102(root=data_dir, split="train", download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
        
        testset = torchvision.datasets.Flowers102(root=data_dir, split="test", download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "pets":
        trainset = torchvision.datasets.OxfordIIITPet(root=data_dir, split="trainval", download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
        
        testset = torchvision.datasets.OxfordIIITPet(root=data_dir, split="test", download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    
    return trainset, testset