import argparse
import time
from collections import OrderedDict
from typing import List

import flwr as fl
import torch.nn.functional as F
import torch_geometric.loader.dataloader
from numpy import ndarray

import common
from utils import *

BATCH_SIZE, TEST_BATCH_SIZE = 512, 512
LR = 0.0005
LOG_INTERVAL = 20


# Define Flower client
class FedDTIClient(fl.client.NumPyClient):

    def __init__(self, model, train, test, cid):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device, non_blocking=True)
        self.batch_size = BATCH_SIZE if len(train) > BATCH_SIZE and len(test) > BATCH_SIZE else min(len(train),
                                                                                                    len(test))
        self.train_loader = torch_geometric.loader.dataloader.DataLoader(train, batch_size=self.batch_size,
                                                                         shuffle=False, num_workers=4)
        self.test_loader = torch_geometric.loader.dataloader.DataLoader(test, batch_size=self.batch_size, shuffle=False,
                                                                        num_workers=4)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        self.id = cid

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        print('Training on {} samples...'.format(len(self.train_loader.dataset)))

        self.model.train()
        epoch = -1
        for batch_idx, data in enumerate(self.train_loader):
            data, target = data.to(self.device, non_blocking=True), data.y.view(-1, 1).float().to(self.device,
                                                                                                  non_blocking=True)
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(data), target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               int(batch_idx * (
                                                                                       len(self.train_loader.dataset) / len(
                                                                                   self.train_loader))),
                                                                               len(self.train_loader.dataset),
                                                                               100. * batch_idx / len(
                                                                                   self.train_loader),
                                                                               loss.item()))

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def get_parameters(self, **kwargs) -> List[ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        loss_mse = 0

        print('Make prediction for {} samples...'.format(len(self.test_loader.dataset)))
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                data, target = data.to(self.device, non_blocking=True), data.y.view(-1, 1).float().to(self.device,
                                                                                                      non_blocking=True)
                output = self.model(data)
                loss_mse += F.mse_loss(output, target, reduction="sum")

        loss = float(loss_mse / len(self.test_loader.dataset))

        return loss, len(self.test_loader.dataset), {"mse": loss}


def main(args):
    model = common.create_model(NORMALISATION)

    if not DIFFUSION:
        train, test = common.load(NUM_CLIENTS, SEED)[args.partition]
    else:
        train, test = common.load(NUM_CLIENTS, SEED, path=FOLDER + DIFFUSION_FOLDER + '/client_' + str(args.partition))

    # Start Flower client
    client = FedDTIClient(model, train, test, args.partition)
    fl.client.start_numpy_client(server_address=args.server, client=client)


start_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--early-stop", default=-1, type=int)
    parser.add_argument("--folder", default=None, type=str)
    parser.add_argument("--seed", type=int, required=True, help="Seed for data partitioning")
    parser.add_argument("--diffusion", action='store_true')
    parser.add_argument("--diffusion-folder", default=None, type=str)
    parser.add_argument("--save-name", default=None, type=str)
    parser.add_argument("--normalisation", default="bn", type=str)
    parser.add_argument(
        "--partition",
        type=int,
        help="Data Partion to train on. Must be less than number of clients",
    )
    parser.add_argument(
        "--server", default='localhost:5050', type=str, help="server address", required=True,
    )
    args = parser.parse_args()

    global NUM_CLIENTS
    global SEED
    global DIFFUSION
    global FOLDER
    global DIFFUSION_FOLDER
    global NORMALISATION
    global DEVICE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLIENTS = args.num_clients
    SEED = args.seed
    DIFFUSION = args.diffusion
    FOLDER = args.folder
    DIFFUSION_FOLDER = args.diffusion_folder
    NORMALISATION = args.normalisation

    main(args)
print("--- %s seconds ---" % (time.time() - start_time))


