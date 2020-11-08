# ING document classification

Repository for training and serving a news classifier.

...

## Repostitory structure

*Only non-obvious files are included below*

    .
    ├── README.md
    ├── data
    │   ├── raw                                     # unzipped .csv dataset
    │   └── zip                                     # zipped dataset
    ├── mlruns                                      # tracking ml runs
    │   └── 1
    │       └── 592dfd384f8243d3a772b1343d1646c0    # exemplary run log dir
    ├── notebooks
    │   ├── exploratory.ipynb                       # eda notebook
    │   └── modeling.ipynb                          # initial modeling notebook
    ├── pipeline.yaml                               # config for training a model
    ├── run_training.py                             # script for running training
    ├── app.py                                      # serving app entry point
    └── src
        ├── api
        │   ├── endpoints.py                        # endpoint prediction training
        │   └── model.py                            # request response models
        ├── ml
        │   ├── eval.py                             # evaluation functionalities
        │   └── train.py                            # training functionalities
        └── utils
            └── utils.py                            # download, preprocessing, mapping

## Setup

How to setup this project.

    $ conda env create -f environment.yaml

### Usage

...

#### Train news classifier

...

#### Serve news classifier

Endpoint for predicting the category of a news title.

    curl -X POST "localhost:8000/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{"title":"apple earnings drop after first quarter"}"
