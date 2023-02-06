"""
 TorchServe uses the concept of handlers to define how requests are processed by a served model.
 A nice feature is that these handlers can be injected by client code when packaging models,
 allowing for a great deal of customization and flexibility.
"""

from abc import ABC
import json
import logging
import os
import torch
from transformers import AutoModelForSequenceClassification
from ts.torch_handler.base_handler import BaseHandler
from preprocessing_tools import PreProcessText

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # Read model serialize/pt file
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        # hard coded for now
        self.preprocess_tool = PreProcessText(**{'tokenization':
                                          {'source': 'huggingface',
                                           'config': {'tokenizer': 'distilbert-base-uncased',
                                                      'auto':True,
                                                      'location': 'transformers',
                                                      'kwargs':{'padding':'max_length',
                                                                'max_length':28,
                                                                'truncation':True,
                                                                'return_tensors':'pt'}
                                                      }},
                               'remove_stopwords': False, 'lowercase': False,
                               'stemming':None,
                               'lemmatization': None})
        self.model.eval()
        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing text
        """
        single_in = data[0]['body']
        single_in = single_in.decode('utf-8')
        logger.info("Received text: '%s'", single_in)
        inputs = self.preprocess_tool([single_in])
        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        with torch.no_grad():
            # Get predictions for both test and valid set
            out = self.model(inputs['input_ids'], inputs['attention_mask'])[0].argmax().item()

        if self.mapping:
            prediction = self.mapping[str(out)]

        return [prediction]


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)
        if data is None:
            return None
        data = _service.preprocess(data)
        data = _service.inference(data)
        return data
    except Exception as e:
        raise e