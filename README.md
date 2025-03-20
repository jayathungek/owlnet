# Barn Owl vocal Individuality Demo
![](./img/banner.jpg)

## What is this?
This codebase is a workshop/demo exploring patterns in a 4-hour barn owl recording. We use spectrogram analysis and a zero-crossing algorithm to isolate individual owl chirps. Since the calls are naturally spaced—possibly due to a feeding negotiation tactic—it’s easy to segment them without overlap. The result is a dataset of around 3.5k distinct chirps, which can be used for clustering and classification. The repo includes code for data loading, feature extraction, and visualisation to analyse vocalisation patterns in a structured way.This is the codebase for a project that was featuresd in an AI for sustainability conference. 
 
## Run it yourself
Please ensure that you have conda and git installed on your system before starting.

1. **Getting files**: Navigate to your working directory. Then:
    ```bash
    git clone https://github.com/jayathungek/owlnet.git
    cd owlnet
    ```
1. **Setting up data directory:** Create a folder named `owl_data` in the root directory of the project (i.e. `owlnet`). If you have a model checkpoint, put it in the `model_checkpoints` folder.
1. **Installing dependencies**
    ```bash
    conda env create -f environment.yml
    conda activate owlnet
    ```
1. **Running the Jupyter notebook**: This will open a browser window from which you can select the notebook you wish to run. If you just want to run the demo, this is `owlnet_demo.ipynb`
    ```bash
    jupyter notebook
    ```

### Optional: Training from scratch 
If you would like to train your own model with different data or a modified architecture, please run `training.ipynb`

## Get the data and model checkpoint
Please contact me [here](mailto:kjayathunge@bournemouth.ac.uk) to request access to the barn owl recordings and model checkpoint. 