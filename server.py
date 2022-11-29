import argparse
import time
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict, Callable

import flwr as fl
import torch.nn.functional as F
import torch_geometric.loader.dataloader
from flwr.common import FitRes, Parameters, Scalar, NDArrays, FitIns, EvaluateIns, MetricsAggregationFn, \
    ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

import common
from utils import *

BATCH_SIZE, TEST_BATCH_SIZE = 512, 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_eval_fn(model) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    test_data = TestbedDataset(root=FOLDER, dataset='kiba' + '_test')
    test_loader = torch_geometric.loader.dataloader.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                               num_workers=4, pin_memory=True)

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        loss_mse = 0
        print('Make prediction for {} samples...'.format(len(test_loader.dataset)))

        with torch.no_grad():
            for data in test_loader:
                data, target = data.to(DEVICE, non_blocking=True), data.y.view(-1, 1).float().to(DEVICE,
                                                                                                 non_blocking=True)
                output = model(data)
                loss_mse += F.mse_loss(output, target, reduction="sum")

            mse = float(loss_mse / len(test_loader.dataset))
        return mse, {'MSE': mse}

    return evaluate


def main(args):
    model = common.create_model(args.normalisation, DEVICE)

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        EARLY_STOP = False

        def __init__(
                self,
                *,
                fraction_fit: float = 1.0,
                fraction_evaluate: float = 1.0,
                min_fit_clients: int = 2,
                min_evaluate_clients: int = 2,
                min_available_clients: int = 2,
                evaluate_fn: Optional[
                    Callable[
                        [int, NDArrays, Dict[str, Scalar]],
                        Optional[Tuple[float, Dict[str, Scalar]]],
                    ]
                ] = None,
                on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                accept_failures: bool = True,
                initial_parameters: Optional[Parameters] = None,
                fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                early_stopping_epochs=5
        ) -> None:
            super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                             min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
                             min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                             on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                             accept_failures=accept_failures, initial_parameters=initial_parameters,
                             fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                             evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
            self.early_stopping = False
            self.epochs_without_improvement = 0
            self.last_better_loss_value = 1000
            self.early_stopping_epochs = early_stopping_epochs
            self.best_model = None
            self.best_metric = None

        def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            aggregated_weights = super().aggregate_fit(server_round, results, failures)
            if aggregated_weights is not None and server_round == args.num_rounds:
                # Save aggregated_weights
                print(f"Saving round {server_round} aggregated_weights...")
                np.savez(f"{args.save_name}.npz", *aggregated_weights)
            return aggregated_weights

        def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            loss, metrics = super().evaluate(server_round, parameters)
            if self.early_stopping_epochs >= 0:
                if loss < self.last_better_loss_value:
                    self.last_better_loss_value = loss
                    self.epochs_without_improvement = 0
                    self.best_model = parameters
                    self.best_metric = metrics
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement > self.early_stopping_epochs:
                        self.early_stopping = True
                        print("EARLY STOPPING TRIGGERED")
                        print(f"Saving aggregated_weights...")
                        weights = parameters_to_ndarrays(self.best_model)
                        np.savez(f"{args.save_name}.npz", *weights)
                        loss = self.last_better_loss_value
                        metrics = self.best_metric
            return loss, metrics

        def configure_fit(
                self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
            fit_list: List[Tuple[ClientProxy, FitIns]] = super().configure_fit(server_round, parameters, client_manager)
            if self.early_stopping:
                fit_list = []
            return fit_list

        def configure_evaluate(
                self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
            evaluate_list: List[Tuple[ClientProxy, EvaluateIns]] = super().configure_evaluate(server_round, parameters,
                                                                                              client_manager)
            if self.early_stopping:
                evaluate_list = []
            return evaluate_list

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=get_eval_fn(model),
        initial_parameters=ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in model.state_dict().items()]),
        early_stopping_epochs=args.early_stop
    )

    fl.server.start_server(strategy=strategy, config=fl.server.ServerConfig(num_rounds=args.num_rounds))


start_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--early-stop", default=-1, type=int)
    parser.add_argument("--folder", default='data/', type=str)
    parser.add_argument("--seed", type=int, required=True, help="Seed for data partitioning")
    parser.add_argument("--diffusion", action='store_true')
    parser.add_argument("--diffusion-folder", default=None, type=str)
    parser.add_argument("--save-name", default=None, type=str)
    parser.add_argument("--normalisation", default="bn", type=str)
    args = parser.parse_args()

    global NUM_CLIENTS
    global SEED
    global DIFFUSION
    global FOLDER
    global DIFFUSION_FOLDER
    global NORMALISATION
    NUM_CLIENTS = args.num_clients
    SEED = args.seed
    DIFFUSION = args.diffusion
    FOLDER = args.folder
    DIFFUSION_FOLDER = args.diffusion_folder
    NORMALISATION = args.normalisation

    main(args)
print("--- %s seconds ---" % (time.time() - start_time))


