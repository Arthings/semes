source ~/export_variables.sh

# Create the job from yaml file.
az ml job create \
    --file install_voc.yaml \
    --subscription $AZUREML_SUBSCRIPTION \
    --resource-group $AZUREML_RESSOURCE_GROUP \
    -w $AZUREML_WORKSPACE_NAME --verbose