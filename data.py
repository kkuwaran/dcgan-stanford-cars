
import multiprocessing
from torchvision import datasets, transforms
import torch



def get_dataloader(root_path: str, image_size: int, batch_size: int):
    """Get dataloader for Stanford Cars dataset."""

    # define transform for dataset
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # get dataset
    dataset_train = datasets.StanfordCars(root=root_path, download=False, split='train', transform=transform)
    dataset_test = datasets.StanfordCars(root=root_path, download=False, split='test', transform=transform)
    dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
    print(f"Size of the dataset: {len(dataset)}")
    
    # get dataloader
    workers = multiprocessing.cpu_count()
    print(f"Using {workers} workers")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, persistent_workers=True if workers > 0 else False,
    )

    return dataloader