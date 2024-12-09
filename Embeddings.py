from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from transformers import BertTokenizer
import pandas as pd
import numpy as np

class Embedders_Five:
    def __init__(self, user_stories):
        """
        Initialize the class with user stories.
        :param user_stories: A column from DataFrame that consist of corpus 
        """
        self.user_stories = user_stories
    
    def getTFIDFEmbeddings(self):
        """
        Compute and return the TF-IDF embeddings.
        Uses CountVectorizer with max_features=100 like in their implementation
        """
        countvec = CountVectorizer(max_features=100)
        bow = countvec.fit_transform(self.user_stories).toarray()
        tfidfconverter = TfidfTransformer()
        X = tfidfconverter.fit_transform(bow).toarray()
        return X
    
    def getBERTEmbeddings(self):
        """
        Get BERT tokenized representations with max_length=100 as in their implementation
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_data = tokenizer(self.user_stories.tolist(), 
                                 padding=True, 
                                 truncation=True, 
                                 max_length=100)
        
        traindata = []
        for msg in tokenized_data['input_ids']:
            traindata.append(msg)
            
        return np.array(traindata)