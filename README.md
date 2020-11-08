# End-to-end news classification

Repository for training and serving a news classifier.

- Uses the [News Aggregator Dataset](https://www.kaggle.com/uciml/news-aggregator-dataset) to train a headline classifier.

- Uses [mlflow](https://www.mlflow.org/docs/latest/index.html) for keeping track of experiments and model management.

- Uses [FastAPI](https://fastapi.tiangolo.com/) to provide simple REST APIs for serving and training models.

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
    ├── run_ml.py                                   # functions for running ml pipeline
    ├── app.py                                      # serving app entry point
    └── src
        ├── api
        │   ├── endpoints.py                        # endpoint prediction training
        │   └── model.py                            # request response models
        ├── ml
        │   ├── eval.py                             # evaluation functionalities
        │   └── train.py                            # training functionalities
        └── utils
            └── utils.py                            # downloading and handling data

## Setup

### Docker (recommended)

The image can be built by cloning this repository and running:

    $ docker build -t ing:latest .

The app can then be started by running:

    $ docker run -d -p 8000:8000 --name ing-service ing

### Anaconda (not recommended)

Alternatively, the project can be setup using [Anaconda](https://www.anaconda.com/) and the provided ```environment.yaml``` by running:

    $ conda create -f environment.yml --name news

The app can be started by running:

    $ uvicorn app:app

## Usage

### Experimentation (optional)

The app provides a command line interface (CLI) for ml experimentation and hyperparametersearch. One possible workflow is the following:

1. Edit the ```steps``` in the ```src.ml.train```module to different models.
2. Edit the ```pipeline.yaml``` to search for the models best hyperparameters.
3. Run ```$ python run_ml.py pipeline.yaml``` to experiment
4. Track experiments using the mlflow Tracking APIs e.g. by running ```$ mlflow ui```

### Train and deploy a news classifier

Assuming the app is running, the first step is to train a classifier using the ```train_and_deploy``` endpoint. Per default, the ```pipeline.yaml``` file is used to perform a gridsearch over hyperparameters before evaluation on the test dataset. Processing time is expected to be < 10 seconds. The trained classifier is then autoamtically deployed to serve the [predict API](#Serve-news-classifier) Optionally, this file can be modified with valid sklearn parameters before(!) starting the service with ```docker run```.

    $ curl -X GET "http://localhost:8000/train_and_deploy" -H  "accept: application/json"

Example response:

```json

    {
        "new model id": "5ae747ac8c274ae6ae2c722040403d79",
        "final hyperparameters": {
            "naivebayes__alpha": 1,
            "tfidf__use_idf": true,
            "vectorizer__max_features": 20000
        },
        "evaluation results": {
            "business": {
            "precision": 0.8956759882969313,
            "recall": 0.908455685719708,
            "f1-score": 0.9020205740520262,
            "support": 57961
            },
            "science and technology": {
            "precision": 0.9497659946513063,
            "recall": 0.9685197325291727,
            "f1-score": 0.9590511925009412,
            "support": 76270
            },
            "entertainment": {
            "precision": 0.9590336649189004,
            "recall": 0.8594703789908217,
            "f1-score": 0.90652647181435,
            "support": 22771
            },
            "health": {
            "precision": 0.898412581352901,
            "recall": 0.8989263577331759,
            "f1-score": 0.8986693961105425,
            "support": 54208
            },
            "accuracy": 0.9224184460963023,
            "macro avg": {
            "precision": 0.9257220573050097,
            "recall": 0.9088430387432196,
            "f1-score": 0.916566908619465,
            "support": 211210
            },
            "weighted avg": {
            "precision": 0.9227415044911694,
            "recall": 0.9224184460963023,
            "f1-score": 0.9222405845306619,
            "support": 211210
            }
        },
        "run information": {
            "artifact_uri": "file:///ing/mlruns/1/5ae747ac8c274ae6ae2c722040403d79/artifacts",
            "end_time": null,
            "experiment_id": "1",
            "lifecycle_stage": "active",
            "run_id": "5ae747ac8c274ae6ae2c722040403d79",
            "run_uuid": "5ae747ac8c274ae6ae2c722040403d79",
            "start_time": 1604862517413,
            "status": "RUNNING",
            "user_id": "root"
        }
    }
```

### Serve the news classifier

After training and deploying a classifier, the ```predict``` endpoint can be used make predictions for new news titles. Note that the endpoint only uses the news title as a feature and accepts only a single string as input, i.e. does not batch predictions (!).

    $ curl -X GET "http://localhost:8000/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"title\":\"apple earnings crash after insider leak\"}"

Example response:

```json
    {
        "title": "apple earnings crash after insider leak",
        "label": "b",
        "category": "business"
    }
```

### Check the docs

Since this app uses FastAPI, visit http://localhost:8000/docs to checkout the API documentation and run requests.
