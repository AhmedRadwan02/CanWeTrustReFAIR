# (define class -> input is the dataset (3) -> define train and tests x_t, y_t, x_T, y_T -> results) *for loop data set (3) * (5)
import numpy as np
import pandas as pd
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import hamming_loss, f1_score


def hamming_loss_score(y_true, y_pred):
    """
    Helper function to calculate hamming loss between true and predicted labels.
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
       
    Returns:
        float: Hamming loss score between 0 and 1
    """
    return hamming_loss(y_true, y_pred)



class ML_Classification:
    """
    A class for training and evaluating multiple machine learning models for classification tasks.
    It leverages LazyPredict to automatically train and evaluate multiple ML models. 
    Will be used in loop: 3 (datasets) x 5 (embedding types per dataset) = 15 (total combinations to evaluate)

    Methods:
        train_ml_models() returns tuple of best performing models
    
    """
    def __init__(self):
        """
        Initializes ML_Classification with 2 LazyPredict classifiers
        
        """
        # 1. Standard classifier for basic evaluation 
        self.lazy_classifier = LazyClassifier(verbose=0, ignore_warnings=True)
        # 2. Classifier with custom hamming loss metric
        self.clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=hamming_loss_score)
    

    def train_ml_models(self, x_train, y_train, x_test, y_test):
        """
        Train multiple models to evaluate thier perofrmance and returns best model name with it's performance metrics.
       
        Params:
            x_train (array-like): Features for training (comes from embeddings)
            y_train (array-like): Target labels for training
            x_test (array-like): Features for testing
            y_test (array-like): Target labels for testing
            
        Returns:
            tuple: (best_model_name, performance_dataframe)
                - 'best_model_name' (string): Name of best performing model
                - performance_df (dataframe): Performance metrics for all models
       """
        
        # Train models using LazyPredict - models_df is first item in tuple
        models_df = self.clf.fit(x_train, x_test, y_train, y_test)[0]       #self.clf.fit returns tuple - we want first element df [0]
        # print(models_df)
        
        # Extract relevant metrics into new DataFrame
        performance_df = models_df[['Accuracy', 'F1 Score', 'hamming_loss_score']].copy()    # Keeping only Accuracy, F1 Score and hamming loss
        performance_df.index.name = 'Model'
        
        # Find best performing model based on accuracy
        best_model_name = performance_df.index[performance_df['Accuracy'].argmax()]
        
        # Return best model name and all performance metrics
        # Reset index to make Model name a column
        return best_model_name, performance_df.reset_index()
         
