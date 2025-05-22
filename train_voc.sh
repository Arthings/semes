source ~/export_variables.sh

python3 -m build ~/libraries/bibcval4
cp ~/libraries/bibcval4/dist/bibcval4-0.0.1-py3-none-any.whl .


# Create the job from yaml file.
az ml job create \
    --file train_voc.yaml \
    --subscription $AZUREML_SUBSCRIPTION \
    --resource-group $AZUREML_RESSOURCE_GROUP \
    -w $AZUREML_WORKSPACE_NAME --verbose

    
# Create the job from yaml file.
az ml job create \
    --file train_voc_entropy.yaml \
    --subscription $AZUREML_SUBSCRIPTION \
    --resource-group $AZUREML_RESSOURCE_GROUP \
    -w $AZUREML_WORKSPACE_NAME --verbose