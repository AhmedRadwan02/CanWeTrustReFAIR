import pandas as pd
from typing import List, Dict
import math
class SensitiveFeaturesMapper:
    def __init__(self, domains_file_path: str, tasks_file_path: str):
        """Initialize the mapper with paths to the CSV files."""
        self.domains_mapping = pd.read_csv(domains_file_path)
        self.tasks_mapping = pd.read_csv(tasks_file_path)
    
    def get_sensitive_features(self, ml_tasks: List[str], domain: str) -> Dict:
        """
        Get sensitive features for given ML tasks and domain.
        """
        intersected_features = {}
        
        # Get domain features using boolean indexing
        domain_features = self.domains_mapping[
            self.domains_mapping['Domain'].str.lower() == domain.lower()
        ]['Feature'].tolist()
        
        # Create dictionary with ML tasks and their features
        ml_tasks_features = {}
        for task in ml_tasks:
            task_features = self.tasks_mapping[
                self.tasks_mapping['Task'].str.lower() == task.lower()
            ]['Feature'].tolist()
            ml_tasks_features[task] = task_features
            
            # Intersect domain and task features
            intersected_features[task] = list(set(ml_tasks_features[task]).intersection(domain_features))

        return intersected_features
    

    def get_unique_sensitive_features(self, ml_tasks: List[str], domain: str) -> set:
        """
        Get the unique sensitive features across all tasks for a given domain.

        Args:
            ml_tasks (List[str]): List of machine learning tasks.
            domain (str): The domain of interest.

        Returns:
            set: A set of unique sensitive features aggregated across tasks.
        """
        # Get task-level intersected sensitive features
        task_sensitive_features = self.get_sensitive_features(ml_tasks, domain)

        # Aggregate features across all tasks and deduplicate them
        unique_sensitive_features = set()
        for features in task_sensitive_features.values():
            unique_sensitive_features.update(features)

        return unique_sensitive_features
