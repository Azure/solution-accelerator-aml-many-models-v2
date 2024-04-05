# Many Models Forecasting with the Azure Machine Learning CLI

The `ml` extension to the Azure CLI is the enhanced interface for Azure Machine Learning. It enables you to train and deploy models from the command line by authoring YAML definition files.

The `cli_pipelines` folder contains 3 different Azure Machine Learning pipeline YAML definition files to train, run inference and evaluate models with a Many Models approach. The pipelines leverage the python source code files in [src](./src)

## Pre-requisites

- Ensure you have access to an Azure subscription
- Install the Azure CLI, instructions [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
- Install the ml extension to the Azure CLI
```az extension add -n ml```
- Create an Azure Machine Learning workspace

## Running the pipelines on Azure Machine Learning through the CLI

1. Log in with the CLI `az login`
2. Configure the CLI to point to your Azure Machine Learning workspace
```
az account set --subscription <subscription ID>
az configure --defaults workspace=<Azure Machine Learning workspace name> group=<resource group>
```
3. Create the compute cluster needed to run the pipelines:
Modify the [compute.yml](./compute.yml) file as needed and run the following command
```
cd cli_pipelines
az ml compute create -f compute.yml
```
**Note:** If you already have an existing cluster you want to use to run the pipelines, modify the `compute` value in the pipeline definition YAML files in the repository

5. Modify the [1_training_pipeline.yml](./1_training_pipeline.yml) and [2_training_pipeline.yml](./2_inference_pipeline.yml) with your Workspace's tracking URI

6. Run the pipelines using the `az ml job create` command
```
az ml job create -f 1_training_pipeline.yml
```

note that you can overwrite the default workspace by adding the `--workspace-name` flag to your CLI commands.

## Helpful links

- [How to create compute targets in Azure Machine Learning using the CLI](https://learn.microsoft.com/en-us/cli/azure/ml/compute?view=azure-cli-latest#az-ml-compute-create)
- [How to use parallel job in  Azure Machine Learning pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?view=azureml-api-2&tabs=cliv2)
