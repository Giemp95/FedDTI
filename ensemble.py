import argparse
import time

import torch.nn.functional as F
import torch_geometric.loader.dataloader

import common
from utils import *

### Super optimized for small RAMs ###

BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(models, args):
    print("Loading test dataset...")
    test_data = TestbedDataset(root="data", dataset='kiba_test')
    test_loader = torch_geometric.loader.dataloader.DataLoader(test_data, batch_size=20000, shuffle=False,
                                                               num_workers=4, pin_memory=True)

    target, output = None, None
    for model_id in range(args.num_models):
        print("Evaluating model " + str(model_id))
        if target is None:
            result = test(models[model_id], target, test_loader)
            output = result[0] / args.num_models
            target = result[1]
        else:
            output_ = test(models[model_id], target, test_loader)
            output += output_ / args.num_models

    return output, target


def test(model, target_flag, test_loader):
    model.eval()
    target_ = None

    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            if target_flag is None:
                data, target = data.to(DEVICE, non_blocking=True), data.y.view(-1, 1).float().to(DEVICE,
                                                                                                 non_blocking=True)
                output = model(data)
                target_ = target
            else:
                data, _ = data.to(DEVICE, non_blocking=True), data.y.view(-1, 1).float().to(DEVICE,
                                                                                            non_blocking=True)
                output = model(data)

    return (output, target_) if target_flag is None else output


def fit(model, opt, epoch, args):
    for model_id in range(args.num_models):
        print("Training model " + str(model_id))
        train(model[model_id], opt[model_id], model_id, epoch, args)


def train(model, opt, model_id, epoch, args):
    print("Loading train dataset...")
    if args.data_folder is None:
        train_data = common.partition(TestbedDataset(root='data', dataset='kiba_train'), args.num_models, args.seed)
    else:
        train_data = TestbedDataset(root='data/' + args.data_folder + '/client_' + str(model_id), dataset='kiba_train')
    train_loader = torch_geometric.loader.dataloader.DataLoader(train_data, batch_size=BATCH_SIZE,
                                                                shuffle=False, num_workers=4, pin_memory=True)

    model.train()
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    for batch_idx, data in enumerate(train_loader):
        data, target = data.to(DEVICE, non_blocking=True), data.y.view(-1, 1).float().to(DEVICE, non_blocking=True)
        opt.zero_grad()
        loss = F.mse_loss(model(data), target)
        loss.backward()
        opt.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           int(batch_idx * (
                                                                                   len(train_loader.dataset) / len(
                                                                               train_loader))),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(
                                                                               train_loader),
                                                                           loss.item()))


def create_model(args):
    model = common.create_model(args.normalisation, DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    return model, opt


def main(args):
    print("Creating models...")
    model, opt = [None] * args.num_models, [None] * args.num_models
    for model_id in range(args.num_models):
        model[model_id], opt[model_id] = create_model(args)

    last_better_loss_value = 1000
    epochs_without_improvement = 0
    early_stopping = False
    for epoch in range(args.num_epochs):
        print("Epoch " + str(epoch))

        if early_stopping:
            break

        fit(model, opt, epoch, args)
        output, target = evaluate(model, args)

        loss = F.mse_loss(output.to("cpu"), target.to("cpu"))

        if args.early_stop >= 0:
            if loss < last_better_loss_value:
                last_better_loss_value = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > args.early_stop:
                    early_stopping = True
                    print("EARLY STOPPING TRIGGERED")
                    print(f"Saving aggregated_weights...")
                    loss = last_better_loss_value
        print("MSE: " + str(loss))


start_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-models", default=1, type=int)
    parser.add_argument("--num-epochs", default=500, type=int)
    parser.add_argument("--early-stop", default=50, type=int)
    parser.add_argument("--folder", default="data", type=str)
    parser.add_argument("--data-folder", default=None, type=str)
    parser.add_argument("--save-name", default=None, type=str)
    parser.add_argument("--normalisation", default="ln", type=str)
    parser.add_argument("--seed", type=int, required=True, help="Seed for data partitioning")
    args = parser.parse_args()

    main(args)
print("--- %s seconds ---" % (time.time() - start_time))

