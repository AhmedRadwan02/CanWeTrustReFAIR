# (define class -> input is the dataset (3) -> define train and tests x_t, y_t, x_T, y_T -> results) * for loop data set (3) * (5)import numpy as np
import pandas as pd
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import hamming_loss, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  

from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset


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
       - train_ml_models() returns tuple of best performing models
       - train_multilabel_ml_models() returns tuple of best performing models
    
    """
    def __init__(self):
        """
        Initializes ML_Classification class with the following components:
            1. LazyPredict Classifiers: A standard LazyClassifier for basic evaluation of models.
            2. Multi-label Models: A dictionary of pre-defined machine learning classifiers for multi-label classification
        
        """
        # 1. Standard classifier for basic evaluation 
        self.lazy_classifier = LazyClassifier(verbose = 0, ignore_warnings = True)
        # 2. Classifier with custom hamming loss metric
        self.clf = LazyClassifier(verbose=0, ignore_warnings = True, custom_metric = hamming_loss_score)
        # 3. Multilabel-Models
        self.models = {
            'RandomForestClassifier': RandomForestClassifier(random_state = 42),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state = 42),
            'GaussianNB': GaussianNB(),
            'LinearSVC': LinearSVC(random_state = 42),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'LogisticRegression': LogisticRegression(random_state = 42)
        }

    def train_ml_models(self, x_train, y_train, x_test, y_test):
        """
        Train multiple models to evaluate thier perofrmance and returns best model name with it's performance metrics.
       
        Params:
            x_train (array-like): Features for training (comes from embeddings)
            y_train (array-like): Target labels for training
            x_test (array-like): Features for testing
            y_test (array-like): Target labels for testing
            
        Returns:
            tuple: (best_model_name, performance_df)
                - 'best_model_name' (string): Name of best performing model based on accuracy
                - performance_df (dataframe): DataFrame containing performance metrics (Accuracy, F1 Score, Hamming Loss) 
                                              for all trained models.
        """
        
        # Train models using LazyPredict - models_df is first item in tuple
        models_df = self.clf.fit(x_train, x_test, y_train, y_test)[0]       #self.clf.fit returns tuple - we want first element df [0]
        # print(models_df)
        
        # Extract relevant metrics into new DataFrame
        performance_df = models_df[['Accuracy', 'F1 Score', 'hamming_loss_score']].copy()    # Keeping only Accuracy, F1 Score and hamming loss
        performance_df.index.name = 'Model'
        
        # Find best performing model based on accuracy
        best_model_name = performance_df.index[performance_df['Accuracy'].argmax()]
        
        # Return the best model's name and the performance DataFrame
        # Reset index to make Model name a column
        return best_model_name, performance_df.reset_index()
         

    def train_multilabel_ml_models(self, x_train, y_train, x_test, y_test):
            """
            Train and evaluate multi-label classification models using three common methods:
                Binary Relevance, Label Powerset, and Classifier Chains for each base classifier.
            
            Params:
                x_train (array-like): Features for training (comes from embeddings)
                y_train (array-like): Target labels for training
                x_test (array-like): Features for testing
                y_test (array-like): Target labels for testing
                
            Returns:
                tuple: (best_model_name, performance_df)
                    - 'best_model_name' (string): Name of best performing model based on the highest F1 score.
                    - performance_df (dataframe): Dataframe summarizing performance metrics (F1 Score and Hamming Loss) 
                                                  for all tested models and methods.    
            """
            
            results = [] # Initialize a list to store performance results

            # For each base classifier
            for name, base_model in self.models.items():
                try:
                    # Method 1 -----------------
                    # Test with Binary Relevance
                    br = BinaryRelevance(classifier=base_model, require_dense=[True,True])
                    br.fit(x_train, y_train)
                    br_pred = br.predict(x_test)
                    
                    # Convert sparse matrix to dense (if needed)
                    if hasattr(br_pred, 'toarray'):
                        br_pred = br_pred.toarray()
                    
                    # Compute evaluation metrics    
                    br_f1 = f1_score(y_test, br_pred, average='micro')
                    br_hl = hamming_loss(y_test, br_pred)
                    
                    # Append results for Binary Relevance
                    results.append({
                        'Model': f'BR_{name}',
                        'Base_Model': name,
                        'Method': 'Binary Relevance',
                        'F1 Score': br_f1,
                        'hamming_loss_score': br_hl
                    })
                    
                    # Method 2 -----------------
                    # Test with Label Powerset
                    lp = LabelPowerset(classifier=base_model, require_dense=[True,True])
                    lp.fit(x_train, y_train)
                    lp_pred = lp.predict(x_test)
                    
                    # Convert sparse matrix to dense (if needed)
                    if hasattr(lp_pred, 'toarray'):
                        lp_pred = lp_pred.toarray()
                    
                    # Compute evaluation metrics
                    lp_f1 = f1_score(y_test, lp_pred, average='micro')
                    lp_hl = hamming_loss(y_test, lp_pred)
                    
                    # Append results for Label Powerset
                    results.append({
                        'Model': f'LP_{name}',
                        'Base_Model': name,
                        'Method': 'Label Powerset',
                        'F1 Score': lp_f1,
                        'hamming_loss_score': lp_hl
                    })
                    
                    # Method 3 -----------------
                    # Test with Classifier Chains
                    cc = ClassifierChain(classifier=base_model, require_dense=[True,True])
                    cc.fit(x_train, y_train)
                    cc_pred = cc.predict(x_test)
                    
                    # Convert sparse matrix to dense (if needed)
                    if hasattr(cc_pred, 'toarray'):
                        cc_pred = cc_pred.toarray()
                    
                    # Compute evaluation metrics
                    cc_f1 = f1_score(y_test, cc_pred, average='micro')
                    cc_hl = hamming_loss(y_test, cc_pred)
                    
                    # Append results for Classifier Chains
                    results.append({
                        'Model': f'CC_{name}',
                        'Base_Model': name,
                        'Method': 'Classifier Chains',
                        'F1 Score': cc_f1,
                        'hamming_loss_score': cc_hl
                    })
                    
                except Exception as e:
                    # Print error and continue if a model or method fails
                    print(f"Error with {name}: {str(e)}")
                    continue

            # Create a performance DataFrame from the results
            performance_df = pd.DataFrame(results)
            
            if not performance_df.empty:
                # Sort by F1 Score to identify the best-performing model
                performance_df.set_index('Model', inplace=True)
                performance_df = performance_df.sort_values('F1 Score', ascending=False)
                best_model_name = performance_df.index[0]
            else:
                 # If no results, return an empty DataFrame and None
                performance_df = pd.DataFrame(columns=['Model', 'Base_Model', 'Method', 'F1 Score', 'hamming_loss_score'])
                best_model_name = None
            
            # Return the best model's name and the performance DataFrame
            return best_model_name, performance_df.reset_index()