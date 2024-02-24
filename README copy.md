# Custom Many Models Forecasting with Azure Machine Learning

## Overview
 Training, deployment, and inference of a high volume of machine learning models is a common use case. One example of this scenario is forecasting, where an organization creates hundres of thousands or millions of independent models to predict volumes or revenues at a hyper-specific level. 
  
 This case comes with challenges in scalability. The ability to execute a machine learning development lifecyle for hundreds of thousands of models requires a parallel and repeatable framework.

## Motivation
The goal of of this repository is to provide a thorough example of a scaleable and repeatable framework for training, deployment, and inference of a 'Many Model' architecture using Azure Machine Learning.

## Notebooks
Follow demo notebooks in order to step through the process of developing and deploying repeatable, parallel, training and inferencing using Azure Machine Learning

## Data
The primary dataset is a subset of the a dataset from the [Dominick's / University of Chicago Booth](https://www.chicagobooth.edu/research/kilts/research-data/dominicks) repository. The dataset includes extra simulated data to simultaneously train many of models on Azure Machine Learning. In this specific subset we will deal with about 33 stores and 3 brands of orange juice. (99 total models)  
  
Access to the full 3,991 store dataset can be found here: [AzureML Open Datasets](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-oj-sales-simulated?tabs=azureml-opendatasets)

## Helpful Links and Documentation
[Managing Pipeline and Component Inputs and Outputs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-inputs-outputs-pipeline?view=azureml-api-2&tabs=cli)  
[Query and Compare Experiment Runs with Azure MLFlow](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments-mlflow?view=azureml-api-2)  
[Azure ML Client Docs](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.mlclient?view=azure-python)  
[MLFlow Client Docs](https://mlflow.org/docs/latest/python_api/mlflow.client.html)  
  
