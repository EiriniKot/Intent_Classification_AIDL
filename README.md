# IntentClassification

Project for NLP course 2022-2023 AIDL.
Using a custom dataset, classify Questions as samples into some categories
Task : Multiclass

This [Notebook](https://colab.research.google.com/drive/18qh9-yzfDO9WKemmgcZQN4rHuLbuje2o?usp=sharing
) includes an implementation with a custom LSTM and pretrained embeddings.
The basic tools for ML where Tensorflow + Tensorflow hub

This [Notebook](https://colab.research.google.com/drive/1lhT-oEr5RdUD8HjpHGWVHHbyyUrxjWgw?usp=sharing
) includes an implementation with huggingface pretrained transformers. The basic tools for ML where Pytorch + HuggingFace

[Here](https://docs.google.com/presentation/d/1BobRXmuEhCJZTBXd8EN5QFzR8-QbFC07GIrZaXQg8ec/edit?usp=sharing) you can find the presentation for this project

You can open one of them and do the trainings.
You can also find results in my gdrive:

Output Files :

[Here](https://drive.google.com/drive/folders/1TQMY_o1vUo3wzhGZDP107kP1BcNB7bf7?usp=sharing) is the lstm and the PretrainedEmb model in tensorflow:

[Here](https://drive.google.com/drive/folders/1APnl9eKgSwPFTp9pR6TDoU87RhykzmuS?usp=sharing) is huggingface transformers models:

[Here](https://drive.google.com/drive/folders/1icVLBMryI-TJYI1RTISAy4eIEKPvZPC9?usp=sharing) are the results metrics plots for transformers and all the metrics in a csv: 

[.mar](https://drive.google.com/file/d/1mOtx-0lGSr2GLJ_shWANBMnQVwrHEHi4/view?usp=sharing)


# Torch Serve

Create a conda environment with python 3.10 (it may also work in other python versions but I have not tested in other versions)

1. Install all the requirements using this command :
```
pip install -r requirements.txt
```

Run this in your current conda env:
conda install -c pytorch torchserve torch-model-archiver torch-workflow-archiver

2. Run this on terminal if you have your model (Plz update with your model path after model/ , the example is with roberta)
```
torch-model-archiver --model-name "roberta_intent" \
                     --version 1.0 --serialized-file ./model/roberta-base-20230208T123209Z-001/roberta-base/pytorch_model.bin \
                     --extra-files "./model/roberta-base-20230208T123209Z-001/roberta-base/config.json, ./model/index_to_name.json, ./src/preprocessing_tools.py" \
                     --handler "./app/handler.py"
```
The previous command will produce a file named eirini_roberta.mar 
that can be understood by TorchServe. 
If you want to change the name of the model change eirini_roberta 
to what you like most. 
3. Then move this mar file into deployment/model-store folder (make sure there is model-store folder)

```
mv roberta_intent.mar ./deployment/model-store 
```

4. Now it is time for torch serve

```
torchserve  --start \
            --model-store  ./deployment/model-store \
            --ts-config ./deployment/config.properties \
            --models irini_bert=roberta_intent.mar
```

5. Finally ping in port and the status should be healthy

```
curl http://localhost:8080/ping
```

# Streamlit

You can use the served model by running 

```
streamlit run app/main.py
```

Finally you can

```
torchserve --stop
```