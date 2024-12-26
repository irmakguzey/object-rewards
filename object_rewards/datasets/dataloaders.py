import hydra
import torch
import torch.utils.data as data


# Script to return dataloaders
def get_dataloaders(cfg):

    dataset = hydra.utils.instantiate(cfg.dataset)

    train_dset_size = int(len(dataset) * cfg.train_dset_split)
    test_dset_size = len(dataset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, test_dset = data.random_split(
        dataset,
        [train_dset_size, test_dset_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_sampler = (
        data.DistributedSampler(train_dset, drop_last=True, shuffle=True)
        if cfg.distributed
        else None
    )
    test_sampler = (
        data.DistributedSampler(test_dset, drop_last=True, shuffle=False)
        if cfg.distributed
        else None
    )  # val will not be shuffled

    train_loader = data.DataLoader(
        train_dset,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        num_workers=cfg.num_workers,
        sampler=train_sampler,
    )
    test_loader = data.DataLoader(
        test_dset,
        batch_size=cfg.batch_size,
        shuffle=test_sampler is None,
        num_workers=cfg.num_workers,
        sampler=test_sampler,
    )

    return train_loader, test_loader, dataset
