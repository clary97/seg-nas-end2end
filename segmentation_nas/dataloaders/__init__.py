from dataloaders import sealer
from torch.utils.data import DataLoader
from torchvision import transforms

def make_data_loader(args, **kwargs):
    print("=> Using Distribued Sampler")
    if args.dataset == 'sealer':
        transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        ])

        datasets = sealer.Sealer(args, 'dataset', 'crop', transform=transform)
        num_class = datasets.NUM_CLASSES
        train_set, val_set = sealer.split_dataset(datasets, 0.8)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = None

        if args.autodeeplab == 'search':
            return train_loader, train_loader, val_loader, test_loader, num_class
        elif args.autodeeplab == 'train':
            return train_loader, val_loader, num_class

    else:
        raise NotImplementedError
