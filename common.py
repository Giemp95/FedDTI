import torch
from models.ginconv import GINConvNet
from utils import *

dataset = 'kiba'


def create_model(normalisation, device="cpu"):
    return GINConvNet(normalisation).to(device)


def partition(dataset, num_partitions, seed):
    length = int(len(dataset) / num_partitions)
    partitions = [length for _ in range(num_partitions - 1)]
    partitions.append(len(dataset) - sum(partitions))
    return torch.utils.data.random_split(dataset, partitions, generator=torch.Generator().manual_seed(seed))


def load(num_partitions, seed, path=None):
    if path is None:
        xy_train, xy_test = TestbedDataset(root='data', dataset=dataset + '_train'), TestbedDataset(root='data',
                                                                                                    dataset=dataset + '_test')
        xy_train = partition(xy_train, num_partitions, seed)
        xy_test = partition(xy_test, num_partitions, seed)

        return list(zip(xy_train, xy_test))
    else:
        xy_train, xy_test = TestbedDataset(root=path, dataset=dataset + '_train'), TestbedDataset(root=path,
                                                                                                  dataset=dataset + '_test')

        return xy_train, xy_test


