import os
import sys

import numpy as np
import pandas as pd
from create_data import create_data_distribution



dir_path = os.getcwd()
clients = int(sys.argv[1])
data_dir = str(sys.argv[2])
num_datasets = int(sys.argv[3])

run_path = data_dir + "/run_" + str(clients) + "_" + str(str(num_datasets)+"0%")
os.makedirs(run_path, exist_ok=True)

print("Loading master dataset...")
dataset_master = pd.read_csv(data_dir + "/kiba_train_master.csv", sep=',', header=0)
if clients > 0:
    print("Loading weak dataset 0...")
    dataset_weak = pd.read_csv(data_dir + "/kiba_train_split_0.csv", sep=',', header=0)
    if num_datasets > 1:
        for dataset in range(num_datasets-1):
            dataset_id = dataset+1
            print("Loading weak " + str(dataset_id) + " dataset...")
            partial_dataset = pd.read_csv(data_dir + "/kiba_train_split_" + str(dataset_id) + ".csv", sep=',', header=0)
            dataset_weak = dataset_weak.append(partial_dataset, ignore_index=True)
    #dataset_weak = dataset_weak.sample(frac=1)
    print("Splitting weak dataset in " + str(clients) + " partitions...")
    dataset_weak = np.array_split(dataset_weak, clients)

print("Creating master train/test split...")
train_partition = dataset_master.sample(frac=0.7)
test_partition = dataset_master.drop(train_partition.index)
client_path = run_path + "/client_0"
os.makedirs(client_path, exist_ok=True)
train_partition.to_csv(client_path + "/kiba_train.csv")
print("New TRAIN dataset split created at " + client_path + "/kiba_train.csv")
test_partition.to_csv(client_path + "/kiba_test.csv")
print("New TEST dataset split created at " + client_path + "/kiba_test.csv")
create_data_distribution(client_path)

if clients > 0:
    for client in range(clients):
        client_id = client+1
        client_path = run_path + "/client_" + str(client_id)
        os.makedirs(client_path, exist_ok=True)

        print("Creating client " + str(client_id) + " train/test split...")
        train_partition = dataset_weak[client].sample(frac=0.7)
        test_partition = dataset_weak[client].drop(train_partition.index)

        train_partition.to_csv(client_path + "/kiba_train.csv")
        print("New TRAIN dataset split created at " + client_path + "/kiba_train.csv")
        test_partition.to_csv(client_path + "/kiba_test.csv")
        print("New TEST dataset split created at " + client_path + "/kiba_test.csv")

        create_data_distribution(client_path)

