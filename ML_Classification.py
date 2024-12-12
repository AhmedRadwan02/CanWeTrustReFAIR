# (define class -> input is the dataset (3) -> define train and tests x_t, y_t, x_T, y_T -> results) *for loop data set (3) * (5)
import numpy as np
import pandas as pd
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import hamming_loss, f1_score
###
class ML_Classification:
    """Classifier for multilabel ML task classification using Label Powerset approach.
        Uses LazyPredict to evaluate multiple models and select the best performing one.
   """
    def __init__(self):
    # Initialize classifier with LazyPredict for model evaluation 
       self.lazy_classifier = LazyClassifier(verbose=0, ignore_warnings=True)
       self.best_model = None
       self.best_model_name = None
       self.models_performance = None
    
    def train_ml_models(self,x_train, y_train, x_test, y_test):
        """
        Train multiple models using LazyPredict and return the best performing one.
       
        Args:
            x_train: Training features (embedded)
            y_train: Training labels 
            x_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Contains:
                - 'best_model_name': Name of best performing model
                - 'best_model_performance': Performance metrics of best model
                - 'all_models_performance': Performance of all tried models
       """
        # Train models using LazyPredict
        models_performance, predictions = self.lazy_classifier.fit(
            x_train, x_test,
            y_train, y_test
        )
        
        self.models_performance = models_performance
       
        # Get best model based on F1 score
        # Convert performance dataframe to dictionary for easier handling
        models_dict = models_performance.to_dict('index')
        
        # Find best model (highest F1 score)
        best_model_name = max(models_dict.items(),key=lambda x: x[1]['Accuracy'])[0]
        
        # Get performance metrics for best model
        best_model_performance = {
            'hamming_loss': hamming_loss(y_test, predictions[best_model_name]),
            'f1_score_micro': f1_score(y_test, predictions[best_model_name], average='micro'),
            'f1_score_macro': f1_score(y_test, predictions[best_model_name], average='macro')
        }
        
        return {
            'best_model_name': best_model_name,
            'best_model_performance': best_model_performance,
            'all_models_performance': models_performance
        }
         
