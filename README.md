# IntentClassification

Project for NLP course 2022-2023 AIDL.
Using a custom dataset, classify Questions as samples into some categories
Task : Multiclass

This Notebook includes an implementation with a custom LSTM and pretrained embeddings
https://colab.research.google.com/drive/18qh9-yzfDO9WKemmgcZQN4rHuLbuje2o?usp=sharing

This Notebook includes an implementation with huggingface pretrained transformers
https://colab.research.google.com/drive/1lhT-oEr5RdUD8HjpHGWVHHbyyUrxjWgw?usp=sharing



# Torch Serve
reference :https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18

Create a conda environment with python 3.10

Install all the requirements

Run this in your current conda env:
conda install -c pytorch torchserve torch-model-archiver torch-workflow-archiver

Run this on terminal
```
torch-model-archiver --model-name "irini_bert" \
                     --version 1.0 --serialized-file ./model/distilbert-base-uncased/pytorch_model.bin \
                     --extra-files "./model/distilbert-base-uncased/config.json, ./model/index_to_name.json, ./src/preprocessing_tools.py" \
                     --handler "./app/handler.py"
```
The previous command will produce a file named eirini_roberta.mar 
that can be understood by TorchServe. 
If you want to change the name of the model change eirini_roberta 
to what you like most. Then move this mar file into deployment/model-store folder

```
mv irini_bert.mar ./deployment/model-store 
```

Now it is time for torch serve

```
torchserve  --start \
            --model-store  ./deployment/model-store \
            --ts-config ./deployment/config.properties \
            --models irini_bert=irini_bert.mar
```

Finally ping in port and the status should be healthy

```
curl http://localhost:8080/ping
```

