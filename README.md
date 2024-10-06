# 12-LeadECGClassification

This repo contains the 12-lead version of the ECG transformer classification started here: https://github.com/yixinli19/heart_disease_classification. All .py files were originally copied from the linked repo. As of 9/23/2024, only data_loader.py has been edited.

## Dataset

Link: https://physionet.org/content/challenge-2021/1.0.3/#files

Link: https://physionet.org/content/mitdb/1.0.0/

## Setup

This project has been tested on python 3.12.6. Using a python virtual environment is the best way to run the code in the Jupyter notebooks. From the root of the repo and run the following command:

```
python3 -m venv venv
```

This will create a new directory called `venv` at the root of the project containing the new virtual environemnt. Run the following command to activate the virtual environemnt.

```
source ./venv/bin/activate
```

Then run the following command to install the necessary libraries:

```
pip install -r requirements.txt
```

When running the Jupyter notebook in VS Code for the first time, you will be prompted for an environemnt. Click Python Environments and choose the virtual environment located in the directory you just created.
