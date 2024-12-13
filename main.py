# Custom modules
from ML_Classification import ML_Classification
from Embeddings import Embedders_Five, ModelDownloader
from load_datasets import LoadDatasets

# Core libraries
import numpy as np
import pandas as pd
from pathlib import Path

# Machine Learning
from sklearn.model_selection import KFold


# installation of embedding models (Save them in the EmbeddingModels folder)
downloader = ModelDownloader()
downloader.download_all_models()

# Load The targets of 3 dataset problems
loader = LoadDatasets()
domain_data = loader.load_domain()
ml_data_binary = loader.load_mltask(format="binary")
ml_data_powerset = loader.load_mltask(format="powerset")

# User stories Loading
user_stories = pd.read_excel("/Users/ahmed/Desktop/CanWeTrustReFAIR/CanWeTrustReFAIR/Dataset/Domain_Classification_Data/Synthetic User Stories.xlsx")
user_stories['Domain'] = user_stories['Domain'].str.lower()

embedder = Embedders_Five(user_stories["User Story"])  # since the USs is shared in the 3 no need to repeat

# Preprocess the USs (X value)
x_embeddings = {
    "GloVE": embedder.getGloVEEmbedding(),
    "TFIDF": embedder.getTFIDFEmbeddings(),
    "Word2Vec": embedder.getWord2VecEmbedding(),
    "FastText": embedder.getFastTextEmbedding(),
    "BERT": embedder.getBERTEmbeddings()
}

y_target = {
    "Domain": domain_data,
    "ML_binary": ml_data_binary,
    "ML_LP": ml_data_powerset 
}

# Create base path for results
base_path = Path("/Users/ahmed/Desktop/CanWeTrustReFAIR/CanWeTrustReFAIR/Evaluation")

# Iterate through embeddings and targets
for emb_name, emb_data in x_embeddings.items():
    for target_name, target_data in y_target.items():
        # Initialize K-Fold Cross-Validation with 10 splits
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Initialize custom classifier
        classifier = ML_Classification()
        
        # Initialize DataFrame to store all folds results
        result_df_fullFolds = pd.DataFrame()
        
        # Performs 10-fold classification
        for fold, (train_index, test_index) in enumerate(kf.split(emb_data), 1):
            x_train, x_test = emb_data[train_index], emb_data[test_index]
            y_train, y_test = target_data[train_index], target_data[test_index]
            
            # Train the models using the classifier defined
            best_model_name, results_df = classifier.train_ml_models(x_train, y_train, x_test, y_test)
            
            # Add only fold number
            results_df["Fold"] = fold
            
            # Concatenate results
            result_df_fullFolds = pd.concat([result_df_fullFolds, results_df], ignore_index=True)
        
        # Determine output directory and filename based on target
        if target_name == "Domain":
            output_dir = base_path / "DomainTaskResults"
        elif target_name == "ML_binary":
            output_dir = base_path / "MLTaskBinResults"
        else:  # ML_LP
            output_dir = base_path / "MLTaskLPResults"
            
        # Create filename with embedding information only
        filename = f"{emb_name}_results.csv"
        output_path = output_dir / filename
        
        # Export results to CSV
        result_df_fullFolds.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        print(f"Processing embedding: {emb_name} for target: {target_name}")