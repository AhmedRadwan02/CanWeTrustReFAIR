import pandas as pd
from typing import List, Dict

class SensitiveFeaturesMapper:
    def __init__(self, domains_file_path: str, tasks_file_path: str):
        """Initialize the mapper with paths to the CSV files."""
        self.domains_mapping = pd.read_csv(domains_file_path)
        self.tasks_mapping = pd.read_csv(tasks_file_path)
    
    def get_sensitive_features(self, ml_tasks: List[str], domain: str) -> Dict:
        """
        Get sensitive features for given ML tasks and domain.
        """
        # Get domain features using boolean indexing
        domain_features = self.domains_mapping[self.domains_mapping['Domain'].str.lower() == domain.lower()]['Feature'].tolist()
        
        # Create dictionary with ML tasks and their features
        ml_tasks_features = {}
        for task in ml_tasks:
            task_features = self.tasks_mapping[self.tasks_mapping['Task'].str.lower() == task.lower()]['Feature'].tolist()
            ml_tasks_features[task] = task_features
        
        return {
            'domain': domain_features,
            'ml_tasks': ml_tasks_features
        }