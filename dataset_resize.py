import torch
import pandas as pd
import numpy as np
import cv2
import os
import pickle

def save_data_as_pickle(X_train, y_train, X_valid, y_valid, X_test, y_test, fname):
    data = {"X_train": X_train, "y_train": y_train, "X_valid": X_valid, "y_valid": y_valid, "X_test": X_test, "y_test": y_test}
    file = open(fname, "wb")
    pickle.dump(data, file)
    file.close()

def load_data_from_pickle(fname):
    file = open(fname, "rb")
    data = pickle.load(file)
    file.close()

    return data

if __name__ == "__main__":
    # Load Data
    print("Loading dataset...")
    data = load_data_from_pickle("data.pkl")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_valid = data["X_valid"]
    y_valid = data["y_valid"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    print("Data loaded!")
    print("Data size:", len(X_train), len(X_valid), len(X_test))

    trainLen = int(len(X_train)/5)
    validLen = int(len(X_valid)/5)
    testLen = int(len(X_test)/5)
    X_train = X_train[:trainLen]
    y_train = y_train[:trainLen]
    X_valid = X_valid[:validLen]
    y_valid = y_valid[:validLen]
    X_test = X_test[:testLen]
    y_test = y_test[:testLen]

    print("Data size:", len(X_train), len(X_valid), len(X_test))

    # print("Data size:", trainLen, validLen, testLen)
    # data = {"X_train": X_train, "y_train": y_train}
    # file = open("data2_train.pkl", "wb")
    # pickle.dump(data, file)
    # file.close()

    # data = {"X_valid": X_valid, "y_valid": y_valid}
    # file = open("data2_valid.pkl", "wb")
    # pickle.dump(data, file)
    # file.close()

    # data = {"X_test": X_test, "y_test": y_test}
    # file = open("data2_test.pkl", "wb")
    # pickle.dump(data, file)
    # file.close()
    # print("Data saved")

    save_data_as_pickle(X_train, y_train, X_valid, y_valid, X_test, y_test, "data_small.pkl")