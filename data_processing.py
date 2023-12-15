import torch
import pandas as pd
import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split

def process_sign_language_MNIST_dataset(rootdir):
    """
    Link to dataset: https://www.kaggle.com/datamunge/sign-language-mnist

    Args: 
        rootdir(str) - path to the MNIST dataset parent folder

    Returns: 
        data (tensor)
        labels (list) 
    """
    
    # Parse data from csv file
    data = pd.read_csv(rootdir)
    data_labels = data["label"]
    data = data.drop(["label"],axis=1)
    data = data.to_numpy()
    data = data.reshape(data.shape[0], 28,28,1)

    # Reformat data labels to lower case letters
    formatted_labels = []
    letters = "abcdefghijklmnopqrstuvwxyz"
    for label in data_labels.values.tolist():
        formatted_labels.append(letters[label])

    # Resize data
    resizedData = []
    for img in data:
        img = np.float32(img)
        resizedImg = cv2.resize(img, (224,224), interpolation = cv2.INTER_CUBIC)
        resizedImg = cv2.cvtColor(resizedImg,cv2.COLOR_GRAY2RGB)
        resizedData.append(resizedImg)

    return torch.tensor(np.array(resizedData)), formatted_labels

def process_massey_gesture_dataset(path_to_files):
    """
    Link to dataset: https://mro.massey.ac.nz/handle/10179/4514
    
    Args: 
        rootdir(str) - path to the Massey parent folder

    Returns: 
        data (list)
        labels (list) 
    
    """
    ret_data = [] 
    ret_label = []
    exclude = ['j','z','0','1','2','3','4','5','6','7','8','9']

    for subdir, dirs, files in os.walk(path_to_files):
        for file in files:
            if "DS_Store" in file:
                continue

            image = os.path.join(subdir, file)           
            if image.split('_')[1] in exclude: 
                continue

            img = cv2.imread(image, cv2.IMREAD_COLOR)
            resized = cv2.resize(img, (224, 224))
            ret_data.append(resized)
            ret_label.append(image.split('_')[1])

    return torch.tensor(np.array(ret_data)), ret_label

def process_fingerspelling_A_dataset(rootdir):
    """
    Link to dataset: https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset

    Args: 
        rootdir(str) - path to the FingerSpelling parent folder

    Returns: 
        data (list)
        labels (list) 
    
    """
    data = []
    labels = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # Check if filename contains DS_Store (for MacOS) or depth (we don't want to keep depth images)
            if "DS_Store" in file or "depth" in file:
                continue
            
            # Store the image and the label
            label = os.path.basename(os.path.normpath(subdir))

            # Get the path of the image and read it
            image_path = os.path.join(subdir, file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # Resized the image as 50x50
            resized_image = cv2.resize(image, (224,224), interpolation = cv2.INTER_CUBIC)
            
            data.append(resized_image)
            labels.append(label) 

    return torch.tensor(np.array(data)), labels

def one_hot_encoding(labels):
    """
    Performs one-hot-encoding on the labels

    Args: 
        labels (list) - dataset labels all in lowercase ie. ['a','b','c']

    Returns: 
        labels_encodings (tensor) - one-hot-encoding representation of the labels
    """

    # Dictionary that stores the letter to integer class conversion
    letterClass = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8,
                    'k':9, 'l':10, 'm':11, 'n':12, 'o':13, 'p':14, 'q':15, 'r':16, 's':17,
                    't':18, 'u':19, 'v':20, 'w':21, 'x':22, 'y':23}
    
    # Convert lowercase letters to numbers
    labelsClass = []
    for label in labels:
        if label in letterClass:
            labelsClass.append(letterClass[label])
        else:
            print("Incorrect label:", label)

    # Get the one-hot-encodings for the labels
    labels_encodings = torch.nn.functional.one_hot(torch.tensor(labelsClass), num_classes=24)
    
    return labels_encodings

def combine_and_split_datasets(datasets, labels, split):
    """
    Combines the given list of datasets, shuffles the data, 
    and splits the combined dataset according to the given split

    Args:
        datasets (list of tensors) - input of dataset
        labels (list of tensors) - labels of dataset (in the same order as datasets)
        split (list of floats) - desired [train, valid, test] split of the combined dataset
                                    (train + valid + test must equal 1)
    
    Returns:
        (X_train, y_train, X_valid, y_valid, X_test, y_test) - tuple of combined and split dataset
    """
    print("Combining datasets...")
    X = torch.cat(datasets, 0)
    y = torch.cat(labels, 0)

    print("Performing train/valid/test split...")
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=split[1]+split[2])
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=split[2]/(split[1]+split[2]))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

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

def load_data():
    print("Processing MNIST...")
    data_mnist, labels_mnist = process_sign_language_MNIST_dataset("datasets/mnist/sign_mnist_data.csv")
    print("Processing MNIST done!")
    
    print("Processing Massey...")
    data_massey, labels_massey = process_massey_gesture_dataset("datasets/massey") # path to massey directory. massey directory has subdir 1,2,3,4,5 
    print("Processing Massey done!")

    print("Processing FingerSpelling...")
    data_fs, labels_fs = process_fingerspelling_A_dataset("datasets/fingerspelling")
    print("Processing FingerSpelling done!")

    # Apply one hot encoding to the labels
    print("Applying one-hot encoding to labels...")
    labels_mnist_encoded = one_hot_encoding(labels_mnist)
    labels_massey_encoded = one_hot_encoding(labels_massey)
    labels_fs_encoded = one_hot_encoding(labels_fs)

    # Combine and split datasets
    X_train, y_train, X_valid, y_valid, X_test, y_test = combine_and_split_datasets([data_mnist, data_massey, data_fs],
                                                                                    [labels_mnist_encoded, labels_massey_encoded, labels_fs_encoded],
                                                                                    [0.8, 0.1, 0.1])
    return X_train, y_train, X_valid, y_valid, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    save_data_as_pickle(X_train, y_train, X_valid, y_valid, X_test, y_test, "data.pkl")
