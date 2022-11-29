import os
import sys

import numpy as np
import pandas as pd

from create_data import create_data_distribution


def normal(mean, var, x):
    return (1 / (np.sqrt(2 * np.pi * var))) * np.exp(-((x - mean) ** 2) / (2 * var))


dir_path = os.getcwd()
clients = int(sys.argv[1])
data_dir = str(sys.argv[2])

dataset_train = pd.read_csv(data_dir + "/kiba_train.csv", sep=',', header=0)
dataset_test = pd.read_csv(data_dir + "/kiba_test.csv", sep=',', header=0)
dataset_complete = dataset_train.append(dataset_test, ignore_index=True).sample(frac=1)

for var in [0.0001, 0.1, 0.2, 0.4, 0.7, 1, 2, 10, 100]:
    values = np.array([normal(0, var, x) for x in np.linspace(-2, 2, clients)])
    values = values / values.sum()
    dist = {}
    for client, value in zip(range(clients), values):
        dist[client] = value
    print("Current distribution for split " + str(clients) + ": " + str(dist))
    continue
    dataset = dataset_complete.copy(deep=True)
    prop_table = dataset['compound_iso_smiles'].value_counts(normalize=True).sample(frac=1)

    run_path = data_dir + "/run_" + str(clients) + "_" + str(var)
    os.makedirs(run_path, exist_ok=True)

    assignments = {}
    dist_sorted = dict(sorted(dist.items(), key=lambda x: x[1]))
    last_client = list(dist_sorted.items())[-1][0]
    for client, perc in dist_sorted.items():
        client_path = run_path + "/client_" + str(client)
        os.makedirs(client_path, exist_ok=True)

        count = 0
        drugs = []
        for drug, prop in prop_table.items():
            if (count + prop) > perc and len(drugs) > 0 and not client == last_client:
                break
            count += prop
            drugs.append(drug)

        prop_table.drop(drugs, inplace=True)

        partition = dataset.loc[dataset['compound_iso_smiles'].isin(drugs)]
        train_partition = partition.sample(frac=0.7)
        test_partition = partition.drop(train_partition.index)

        train_partition.to_csv(client_path + "/kiba_train.csv")
        print("New TRAIN dataset split created at " + client_path + "/kiba_train.csv")
        test_partition.to_csv(client_path + "/kiba_test.csv")
        print("New TEST dataset split created at " + client_path + "/kiba_test.csv")

        create_data_distribution(client_path)


