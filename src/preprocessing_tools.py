# from transformers import AutoTokenizer
import transformers, tokenizers
import wget
import nltk
from nltk import tokenize
from nltk import stem
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.preprocessing import OneHotEncoder
import os
import pandas as pd
import numpy as np
import torch

def one_hot_encoding(non_encoded):
    """
    Simple one hot encoding
    :param non_encoded:
    :return: pd.Series
    """
    lb = OneHotEncoder()
    onehot_labels = lb.fit_transform(non_encoded.values.reshape(-1, 1))
    onehot_encoding = pd.Series(np.array(onehot_labels.toarray()).tolist())
    return onehot_encoding , lb


class PreProcessText:
    def __init__(self,
                 tokenization,
                 stemming,
                 lemmatization,
                 remove_stopwords=True,
                 lowercase=True):

        self.tokenization_config = tokenization['config']
        if 'kwargs' in self.tokenization_config:
            self.tokenizer_kwargs = self.tokenization_config['kwargs']
        else:
            self.tokenizer_kwargs = {}

        self.source = tokenization['source']
        self.tokenizer = getattr(self, self.source +'_tokenizers')(self.tokenization_config['tokenizer'])
        self.func_list = []

        if stemming:
            self.stemmer = getattr(self, stemming['source'] + '_stemmers')(stemming['stemmer'])
            self.func_list.append(self.stemmer)
        if lemmatization:
            self.lemmatizer = getattr(self, lemmatization['source'] + '_lemmatizers')(lemmatization['lemmatizer'])
            self.func_list.append(self.lemmatizer)
        if remove_stopwords and lowercase:
            print('Remove stop and lower')
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            self.func_list.append(lambda i: self.remove_stop(i, lowercase))

    def huggingface_tokenizers(self, tokenizer_name):
        """
        This is a complex routine which tries to capture some tokenization functionality of hugging face.
        If location requested is transformers then there are 2 options.
        1. By setting 'auto' to True the AutoTokenizer module will be used and pass to the function from_pretrained
            the name of the tokenizer aka 'tokenizer_name'
        2. By setting 'auto' to False the module of tokenizer will be called from the transformers. For example if
            tokenizer_name is 'BertTokenizer' then transformers.BertTokenizer will be initialized. The function for
            tokenization will be tokenize
        If location requested is tokenizers then similarly to previous case 2, if tokenizer_name is 'BertWordPieceTokenizer'
        then tokenizers.BertWordPieceTokenizer will be initialized.

        :param tokenizer_name: str
        :return: tokenizer
        """
        if self.tokenization_config['location'] == 'transformers':
            # In case transformers module is requested
            if self.tokenization_config['auto']:
                tokenizer = lambda i: transformers.AutoTokenizer.from_pretrained(tokenizer_name)(i, **self.tokenizer_kwargs)
            else:
                if 'vocabulary' not in self.tokenization_config:
                    raise Exception("""
                                     Please provide a proper vocabulary 
                                    """)
                tokenizer = getattr(transformers, tokenizer_name).from_pretrained(self.tokenization_config['vocabulary']).tokenize
        elif self.tokenization_config['location'] == 'tokenizers':
            if 'vocabulary' not in self.tokenization_config:
                raise Exception("""
                                 Please provide a proper vocabulary 
                                 Examples:
                                 Bert vocabularies
                                 Bert Base Uncased Vocabulary
                                 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
                                 Bert Base Cased Vocabulary
                                 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt
                                 Bert Large Cased Vocabulary
                                 https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt
                                 Bert Large Uncased Vocabulary
                                 https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
                                 GPT-2 Vocabulary
                                 https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
                                 GPT-2 Medium Vocabulary
                                 https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json
                                 GPT-2 Large Vocabulary
                                 https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json """)
            else:
                if not os.path.exists(self.tokenization_config['vocabulary']):
                    wget.download(self.tokenization_config['vocabulary'])
                vocab = self.tokenization_config['vocabulary'].split('/')[-1]
            tokenizer = getattr(tokenizers, tokenizer_name)(vocab).encode
        print(f'{tokenizer} was instanciated')
        return tokenizer

    def nltk_tokenizers(self, tokenizer_name):
        tokenizer = getattr(tokenize, tokenizer_name)(**self.tokenizer_kwargs).tokenize
        print(f'{tokenizer} was instanciated')
        return tokenizer

    def nltk_stemmers(self, stemmer_name):
        stemmer = getattr(stem, stemmer_name)().stem
        print(f'{stemmer} was instanciated')
        return stemmer

    def nltk_lemmatizers(self, lemmatizer_name):
        lemmatizer_name = getattr(stem, lemmatizer_name)().lemmatize
        print(f'{lemmatizer_name} was instanciated')
        return lemmatizer_name

    def tokenization(self, set):
        if isinstance(set, list):
            newset = self.tokenizer(set)
        else:
            newset = set.apply(lambda i: self.tokenizer(i))
        return newset

    def remove_stop(self, token, lowercase=True):
        """
        This function can remove stopwords and apply lower to all tokens.
        :param token:
        :param lowercase:
        :return:
        """
        lower_t = token.lower()
        if lower_t not in self.stop_words:
            if lowercase:
                return lower_t
            else:
                return token

    def token_preprocess(self, tokens):
        """
        This function is responsible for applying all the
        steps for token preprocessing. The steps are defined
        based on config passed in the initialization of the class.
        :param token:
        :return:
        """
        def aux(token):
            for step_func in self.func_list:
                token = step_func(token)
            return token
        tokenized_filtered = list(map(lambda i: aux(i), tokens))
        tokenized_filtered = list(filter(lambda element: element is not None, tokenized_filtered))
        return tokenized_filtered

    def __call__(self, input_data):
        """

        :param input_data:
        :return:
        """
        tokenized_sentences = self.tokenization(input_data)
        if isinstance(tokenized_sentences, transformers.tokenization_utils_base.BatchEncoding):
            return tokenized_sentences

        sample1 = tokenized_sentences.iloc[0]
        # check encoding
        if not isinstance(sample1, tokenizers.Encoding):
            tokenized_filtered = list(map(lambda sent: self.token_preprocess(sent), tokenized_sentences))
        else:
            tokenized_filtered = tokenized_sentences
        return tokenized_filtered


class DatasetBuilder(torch.utils.data.Dataset):
    def __init__(self, input_x, target_y):
        """
        A simple dataset builder that we can pass to the Trainer
        :param input_x: dict
        :param target_y: list
        """
        self.input_x = input_x
        self.target_y = target_y
        self.keys = input_x.keys()

    def __getitem__(self, idx):
        item ={key: val[idx] for key, val in self.input_x.items()}
        item['label'] = self.target_y[idx]
        return item

    def __len__(self):
        return len(self.target_y)