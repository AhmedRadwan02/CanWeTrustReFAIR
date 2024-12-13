# Standard library imports
import os
from pathlib import Path
import urllib.request
import zipfile

# Data processing imports
import numpy as np
import pandas as pd

# NLP imports
import nltk
from nltk.tokenize import word_tokenize
import fasttext
import fasttext.util

# Gensim imports
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.downloader as api

# Sklearn imports
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer
)

# Transformers imports
from transformers import BertTokenizer

class ModelDownloader:
    def __init__(self, base_path="/Users/ahmed/Desktop/CanWeTrustReFAIR/CanWeTrustReFAIR/EmbeddingModels"):
        """Initialize model downloader with base path for storing models."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Define model paths
        self.fastext_path = self.base_path / "cc.en.300.bin"
        self.word2vec_path = self.base_path / "word2vec-google-news-300.bin"
        self.glove_zip = self.base_path / "glove.6B.zip"
        self.glove_txt = self.base_path / "glove.6B.100d.txt"
        self.glove_word2vec = self.base_path / "glove.6B.100d.word2vec"
        
    def download_fasttext(self):
        """Download fastText model if not exists."""
        if not self.fastext_path.exists():
            print("Downloading fastText model...")
            fasttext.util.download_model('en', if_exists='ignore')
            # Move the downloaded file to the correct location if needed
            default_path = Path("cc.en.300.bin")
            if default_path.exists():
                default_path.rename(self.fastext_path)
            print("fastText model downloaded successfully")
        else:
            print("fastText model already exists")

    def download_word2vec(self):
        """Download Word2Vec model if not exists."""
        if not self.word2vec_path.exists():
            print("Downloading Word2Vec model...")
            word2vec_model = api.load('word2vec-google-news-300')
            word2vec_model.save_word2vec_format(str(self.word2vec_path), binary=True)
            print("Word2Vec model downloaded and saved successfully")
        else:
            print("Word2Vec model already exists")

    def download_glove(self):
        """Download and process GloVe vectors if not exists."""
        glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
        
        if not self.glove_zip.exists():
            print("Downloading GloVe vectors...")
            urllib.request.urlretrieve(glove_url, str(self.glove_zip))
            
            print("Extracting GloVe vectors...")
            with zipfile.ZipFile(self.glove_zip, 'r') as zip_ref:
                zip_ref.extractall(str(self.base_path))
            print("GloVe vectors extracted successfully")
        
        if not self.glove_word2vec.exists():
            print("Converting GloVe to Word2Vec format...")
            glove2word2vec(str(self.glove_txt), str(self.glove_word2vec))
            print("Conversion to Word2Vec format complete")
        else:
            print("GloVe vectors already exist in Word2Vec format")

    def download_all_models(self):
        """Download all required models."""
        print("Starting download of all models...")
        self.download_fasttext()
        self.download_word2vec()
        self.download_glove()
        print("All models downloaded successfully")

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