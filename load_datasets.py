import os
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain

class LoadDatasets:
    def __init__(self):
        self.domain_path = "./Dataset/Domain_Classification_Data/Synthetic User Stories.xlsx"
        self.ml_path = "./Dataset/ML_Tasks_Classification_Data/Synthetic User Stories.xlsx"
        self.labels_path = "./Dataset/ML_Tasks_Classification_Data/Keyword labelled.xlsx"

    def load_domain(self):
        dataset = pd.read_excel(self.domain_path)
        target = []
        for row in dataset.iterrows():
            target.append(np.where(dataset["Domain"].unique() == row[1]["Domain"])[0][0])
        dataset["Target"] = target
        y = dataset['Target']
        return pd.DataFrame(y)

    def load_mltask(self):
        dataset = pd.read_excel(self.ml_path)
        labels = pd.read_excel(self.labels_path, header=None)
        
        labels[2] = labels[2].apply(lambda x: x.lower())
        categories_column = []
        for row in labels.iterrows():
            current_labels = []
            for label in row[1][3:]:
                if isinstance(label, str):
                    current_labels.append(label.lower())
            categories_column.append(current_labels)
        labels["Categories array"] = categories_column
        target = []
        for row in dataset.iterrows():
            target.append(labels[labels[2] == row[1]["Machine Learning Task"].lower()]
                          ["Categories array"].values[0])
        dataset["Target"] = target
        dataset['Target'] = dataset['Target'].apply(lambda x: ast.literal_eval(str(x)))
        multilabel = MultiLabelBinarizer()
        y = multilabel.fit_transform(dataset['Target'])
        
        return pd.DataFrame(y, columns=multilabel.classes_)