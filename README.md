# ECE324-ASL-Recognition
### Downloading the Dataset
The dataset for this project can be downloaded [here](https://drive.google.com/drive/folders/1kbBPhVgaCz-ZtiUsv3jkCS9KitTPcF7Y?usp=sharing).  
  
To properly set up the required file structure, unzip `dataset.zip` and place this folder in the main directory of the repo. This is necessary for running `data_processing.py` later on and contains the raw, unprocessed versions of each of the three chosen datasets.
  
Alternatively, if you wish to use the pre-processed dataset and skip the running of `data_processing.py`, unzip the file `data.pkl.zip` and place it in the main directory of the repo before running `train.py`. Please note that `data.pkl` is a very large file (around 22 GB).

### Data Preprocessing
The data processing of the pipeline is performed in `data_processing.py`.
  
This file processes each of the three datasets by resizing the images and converting from greyscale to RGB where needed. It also concatenates the three datasets together to create a combined dataset and performs a training/validation/test split. The processed dataset is finally saved into a pickle file to be used by `train.py`.

### Training Loop
The training of the model is implemented and performed in `train.py`.
  
The training loop currently uses minibatch gradient descent to optmize the weights of the model.

### Results
The results of the first three trials can be seen in the `results_trial1`, `results_trial2`, and `results_trial3` directories.
