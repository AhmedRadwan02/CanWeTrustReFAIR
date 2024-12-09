from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize
import fasttext
import fasttext.util
class Embedders_Five:
    def __init__(self, user_stories):
        """
        Initialize the class with user stories.
        :param user_stories: A column from DataFrame that consist of corpus 
        """
        self.user_stories = user_stories
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        

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
    
    def getWord2VecEmbedding(self):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('word2vec-google-news-300.bin', binary=True)
        traindata = []
        
        for msg in self.user_stories:
            words = word_tokenize(msg.lower())
            vecs = []
            for word in words:
                if word in w2v_model:
                    # make it 100, since it is 300 embedding model
                    vecs.append(w2v_model[word][:100])
            if vecs:
                vec_avg = sum(vecs) / len(vecs)
            else:
                vec_avg = [0] * 100
            traindata.append(vec_avg)
            
        return np.array(traindata)
    

    def getGloVEEmbedding(self):
        # Load GloVe vectors - note the binary=False since GloVe is in text format
        glove_model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec', binary=False)
        traindata = []
        
        for msg in self.user_stories:
            words = word_tokenize(msg.lower())
            vecs = []
            for word in words:
                if word in glove_model:
                    vecs.append(glove_model[word]) 
            if vecs:
                vec_avg = sum(vecs) / len(vecs)
            else:
                vec_avg = [0] * 100  # Zero vector for words not found
            traindata.append(vec_avg)
            
        return np.array(traindata)
    
    def getFastTextEmbedding(self):
        """
        Get FastText embeddings with dimensionality reduction.
        Note: First time running might be slower due to model loading.
        """
        
        # Load the model
        ft = fasttext.load_model('cc.en.300.bin')
        # Reduce dimensions to 100
        fasttext.util.reduce_model(ft, 100)
        traindata = []
        
        for msg in self.user_stories:
            # Get sentence vector directly 
            vec = ft.get_sentence_vector(msg)
            traindata.append(vec)
        
        traindata = np.array(traindata)
        return traindata