import pandas as pd
from typing import Set
import ast
import numpy as np
from collections import defaultdict
import json

def analyze_misclassification(row) -> dict:
    """helper function to analyze if domain and tasks were misclassified."""
    def parse_tasks(task_str: str) -> Set[str]:
        # Turn the tasks string into a set of task names
        try:
            if pd.isna(task_str) or task_str == '{}':
                return set()
            tasks_dict = ast.literal_eval(task_str.replace("'", '"'))
            return set(tasks_dict.keys())
        except:
            return set()
        
    # Get domain and task info from the row
    oracle_domain = row['Oracle Domain']
    refair_domain = row['ReFAIR Domain'] 
    oracle_tasks = parse_tasks(row['Oracle Tasks & Sensitive Features'])
    refair_tasks = parse_tasks(row['ReFAIR Tasks & Sensitive Features'])
    
    return {
        'domain_match': oracle_domain == refair_domain,
        'task_match': oracle_tasks == refair_tasks,
        'oracle_tasks': oracle_tasks,
        'refair_tasks': refair_tasks,
        'extra_tasks': refair_tasks - oracle_tasks,  # Tasks ReFAIR predicted but weren't in oracle
    }

def analyze_predictions(file_path: str):
    """Main function to analyze ReFAIR's predictions."""
    # Load the data
    df = pd.read_csv(file_path)
    
    def parse_features(x):
        """Helper to parse feature sets from the data."""
        try:
            if pd.isna(x) or x == '{}' or x == 'set()':
                return set()
            return set(ast.literal_eval(x))
        except:
            return set()

    # Initialize all my counters
    total_stories = len(df)
    perfect_matches = 0  # Total perfect matches (both empty + with features)
    perfect_matches_both_empty = 0  # Perfect matches where both oracle and ReFAIR had no features
    perfect_matches_with_features = 0  # Perfect matches where at least one had features
    diff_by_one = 0
    diff_by_two = 0
    diff_by_more = 0
    empty_oracle = 0
    empty_refair = 0
    both_empty = 0
    non_empty_perfect = 0  # Track perfect matches where oracle had features

    # Track misclassifications
    wrong_domains = defaultdict(int)  # Count times each domain was wrong
    wrong_tasks = defaultdict(int)  # Count times each task was wrongly added
    
    # Store cases for analysis
    error_cases = []  # All error cases
    domain_errors = []  # Domain misclassifications
    task_errors = []  # Task misclassifications
    
    # Store metrics
    jaccard_scores = []
    overlap_scores = []
    mojo_scores = []
    levenshtein_distances = []
    
    # Analyze each user story
    for idx, row in df.iterrows():
        # Get feature sets
        oracle_features = parse_features(row['Oracle Unique Sensitive Features'])
        refair_features = parse_features(row['ReFAIR Unique Sensitive Features'])
        
        # Check domain and task predictions
        misclass_info = analyze_misclassification(row)
        if not misclass_info['domain_match']:
            domain_errors.append(row)
            wrong_domains[row['ReFAIR Domain']] += 1
        
        if not misclass_info['task_match']:
            task_errors.append(row)
            for task in misclass_info['extra_tasks']:
                wrong_tasks[task] += 1
        
        # Track empty sets
        if not oracle_features:
            empty_oracle += 1
        if not refair_features:
            empty_refair += 1
        if not oracle_features and not refair_features:
            both_empty += 1
            perfect_matches_both_empty += 1
            perfect_matches += 1
            
        # Calculate similarity metrics
        intersection = len(oracle_features.intersection(refair_features))
        union = len(oracle_features.union(refair_features))
        
        # Handle empty set cases
        if not oracle_features and not refair_features:
            jaccard = overlap = 1.0
        else:
            jaccard = intersection / union if union != 0 else 0
            overlap = intersection / max(len(oracle_features), len(refair_features)) if oracle_features or refair_features else 0
        
        # Calculate my distances
        mojo = 1 - ((jaccard + overlap) / 2)
        levenshtein = len(set(sorted(list(oracle_features))).symmetric_difference(set(sorted(list(refair_features)))))
        
        # Store metrics
        jaccard_scores.append(jaccard)
        overlap_scores.append(overlap)
        mojo_scores.append(mojo)
        levenshtein_distances.append(levenshtein)
        
        # Categorize the differences
        if oracle_features or refair_features:  # At least one has features
            if levenshtein == 0:
                perfect_matches += 1
                perfect_matches_with_features += 1
                if len(oracle_features) > 0:  # Oracle had features
                    non_empty_perfect += 1
            elif levenshtein == 1:
                diff_by_one += 1
                error_cases.append((row, 'one_feature'))
            elif levenshtein == 2:
                diff_by_two += 1
                error_cases.append((row, 'two_features'))
            else:
                diff_by_more += 1
                error_cases.append((row, 'more_features'))

    # Print all my results
    print(f"\n=== Overall Performance Analysis ===")
    print(f"Total User Stories: {total_stories}")
    
    print(f"\nSet Statistics:")
    print(f"Empty Oracle Sets: {empty_oracle} ({empty_oracle/total_stories*100:.2f}%)")
    print(f"Empty ReFAIR Sets: {empty_refair} ({empty_refair/total_stories*100:.2f}%)")
    print(f"Both Empty Sets: {both_empty} ({both_empty/total_stories*100:.2f}%)")
    
    print(f"\nPerfect Match Analysis:")
    print(f"Total Perfect Matches: {perfect_matches} ({perfect_matches/total_stories*100:.2f}%)")
    print(f"- Perfect Matches (Both Empty): {perfect_matches_both_empty} ({perfect_matches_both_empty/total_stories*100:.2f}%)")
    print(f"- Perfect Matches (With Features): {perfect_matches_with_features} ({perfect_matches_with_features/total_stories*100:.2f}%)")
    
    print(f"\nError Case Analysis:")
    print(f"Total Error Cases: {len(error_cases)} ({len(error_cases)/total_stories*100:.2f}%)")
    print(f"- Differ by 1 feature: {diff_by_one} ({diff_by_one/total_stories*100:.2f}%)")
    print(f"- Differ by 2 features: {diff_by_two} ({diff_by_two/total_stories*100:.2f}%)")
    print(f"- Differ by more features: {diff_by_more} ({diff_by_more/total_stories*100:.2f}%)")
    
    print(f"\nDistance Metrics:")
    print(f"Mean Jaccard Similarity: {np.mean(jaccard_scores):.4f}")
    print(f"Mean Set Overlap: {np.mean(overlap_scores):.4f}")
    print(f"Mean MoJo Distance: {np.mean(mojo_scores):.4f}")
    print(f"Mean Levenshtein Distance: {np.mean(levenshtein_distances):.4f}")
    
    print(f"\nDomain Classification Analysis:")
    print(f"Domain Errors: {len(domain_errors)} ({len(domain_errors)/total_stories*100:.2f}%)")
    print("\nTop 5 Most Common Domain Misclassifications:")
    for domain, count in sorted(wrong_domains.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {domain}: {count} times")
        
    print(f"\nTask Classification Analysis:")
    print(f"Task Errors: {len(task_errors)} ({len(task_errors)/total_stories*100:.2f}%)")
    print("\nTop 5 Most Common Extra Tasks Predicted:")
    for task, count in sorted(wrong_tasks.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {task}: {count} times")

    # Analyze non-empty oracle cases
    non_empty_oracle = total_stories - empty_oracle
    print(f"\nAnalysis for non-empty oracle cases ({non_empty_oracle} cases):")
    print(f"Perfect matches: {non_empty_perfect} ({non_empty_perfect/non_empty_oracle*100:.2f}%)")



if __name__ == "__main__":
    file_path = "./Dataset/SensitiveFeaturesValidation_Data/overall-oracle-prediciton.csv"
    analyze_predictions(file_path)