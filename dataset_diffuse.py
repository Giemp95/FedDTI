import os
import sys

import numpy as np
import pandas as pd

from create_data import create_data_distribution

TEST = False


# Simple function for obtaining the value of a normal distribution in a certain point
def normal(mean, var, x):
    return (1 / (np.sqrt(2 * np.pi * var))) * np.exp(-((x - mean) ** 2) / (2 * var))


# Dataset reading
dir_path = os.getcwd()
clients = int(sys.argv[1])
data_dir = str(sys.argv[2])

dataset_train = pd.read_csv(data_dir + "/kiba_train.csv", sep=',', header=0)
dataset_test = pd.read_csv(data_dir + "/kiba_test.csv", sep=',', header=0)
dataset = dataset_train.append(dataset_test, ignore_index=True)
drugs_label = 'target_sequence'

# Bucket creation
prop_table = dataset[drugs_label].value_counts(normalize=True).sample(frac=1)
bucket_dist = [1 / clients for _ in range(clients)]
last_client = clients - 1
buckets = {}
for client, perc in enumerate(bucket_dist):
    count = 0
    drugs = []
    for drug, prop in prop_table.items():
        if (count + prop) > perc and len(drugs) > 0 and not client == last_client:
            break
        count += prop
        drugs.append(drug)
    prop_table.drop(drugs, inplace=True)

    partition = dataset.loc[dataset[drugs_label].isin(drugs)]
    buckets[client] = partition

# Creating different datasets based on the Diffusion method
for var in [0.4]:#[0.0001, 0.1, 0.2, 0.4, 0.7, 1, 2, 10, 100]:

    # Distributions calculation
    values = np.array([normal(0, var, x) for x in np.linspace(-2, 2, clients)])
    values = values / values.sum()
    print("Current distribution for split " + str(clients) + ": " + str(values))

    if not TEST:
        run_path = data_dir + "/run_" + str(clients) + "_" + str(var)
        os.makedirs(run_path, exist_ok=True)

    # Create assignments
    assignments = []
    for i in range(clients):
        current = []
        for j in range(clients):
            current.append(values[(j + i) % clients])
        assignments.append(current)

    # Assign data
    data = {}
    for client in range(clients):
        data[client] = pd.DataFrame(columns=dataset.columns)

    for assignment in assignments:
        max_index = assignment.index(max(assignment))
        for drug_name in buckets[max_index][drugs_label].unique():
            drug = buckets[max_index].loc[buckets[max_index][drugs_label] == drug_name].sample(frac=1)

            chunks = []
            previous = 0
            # The +0.5 is useful to equilibrate better the data chunks
            for elem in [int((perc * len(drug)) + 0.5) for perc in assignment]:
                chunks.append(elem + previous)
                previous += elem
            for client, split in enumerate(np.split(drug, chunks[:-1])):
                data[client] = data[client].append(split, ignore_index=True)

    # Function for testing the effective distribution of the drugs
    if TEST:
        for client_ in range(clients):
            total_count = np.array([0. for _ in range(clients)])
            for drug in buckets[client_][drugs_label].unique():
                count = []
                for client in range(clients):
                    if drug in data[client][drugs_label].unique():
                        count.append(data[client][drugs_label].value_counts()[drug])
                    else:
                        count.append(0)
                total_count += (np.array(count) / np.array(count).sum())
            print(total_count / len(buckets[client_][drugs_label].unique()))
        continue

    # Train/test split and conversion to graph
    for client in range(clients):
        train_partition = data[client].sample(frac=0.7)
        test_partition = data[client].drop(train_partition.index)
        if not TEST:
            client_path = run_path + "/client_" + str(client)
            os.makedirs(client_path, exist_ok=True)
            train_partition.to_csv(client_path + "/kiba_train.csv")
            print("New TRAIN dataset split created at " + client_path + "/kiba_train.csv")
            test_partition.to_csv(client_path + "/kiba_test.csv")
            print("New TEST dataset split created at " + client_path + "/kiba_test.csv")

            create_data_distribution(client_path)


