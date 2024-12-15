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

# Installation of embedding models (Save them in the EmbeddingModels folder)
downloader = ModelDownloader()
downloader.download_all_models()

# Load The targets of all dataset problems
loader = LoadDatasets()
domain_data = loader.load_domain()
ml_task_data = loader.load_mltask()  

# User stories Loading
user_stories = pd.read_excel("./Dataset/Domain_Classification_Data/Synthetic User Stories.xlsx")
user_stories['Domain'] = user_stories['Domain'].str.lower()
embedder = Embedders_Five(user_stories["User Story"])

# Get embeddings
x_embeddings = {
    "GloVE": np.array(embedder.getGloVEEmbedding()),
    "TFIDF": np.array(embedder.getTFIDFEmbeddings()),
    "Word2Vec": np.array(embedder.getWord2VecEmbedding()),
    "FastText": np.array(embedder.getFastTextEmbedding()),
    "BERT": np.array(embedder.getBERTEmbeddings())
}

y_target = {
    "Domain": np.array(domain_data),
    "ML_Task": np.array(ml_task_data)
}

# Create base path for results
base_path = Path("./Evaluation")
# Create directories if they don't exist
for dir_name in ["DomainTaskResults", "MLTaskResults"]:
    (base_path / dir_name).mkdir(parents=True, exist_ok=True)

# Iterate through embeddings and targets
for emb_name, emb_data in x_embeddings.items():
    for target_name, target_data in y_target.items():
        # Initialize K-Fold Cross-Validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Initialize classifier
        classifier = ML_Classification()
        
        print(f"\nProcessing embedding: {emb_name} for target: {target_name}")
        
        if target_name == "Domain":
            # Initialize DataFrame for domain classification
            result_df_fullFolds = pd.DataFrame(
                columns=["Fold", "Model", "Accuracy", "F1 Score", "hamming_loss_score", "Best_Model"]
            )
        else:
            # Initialize DataFrame for multi-label classification
            result_df_fullFolds = pd.DataFrame(
                columns=["Fold", "Model", "Base_Model", "Method", "F1 Score", "hamming_loss_score", "Best_Model"]
            )

        # Performs 10-fold classification
        for fold, (train_index, test_index) in enumerate(kf.split(emb_data), 1):
            print(f"Processing Fold {fold}...")
            
            # Split data
            if isinstance(emb_data, pd.DataFrame):
                x_train = emb_data[emb_data.index.isin(train_index)]
                x_test = emb_data[emb_data.index.isin(test_index)]
            else:
                x_train, x_test = emb_data[train_index], emb_data[test_index]
           
            if isinstance(target_data, pd.DataFrame):
                y_train = target_data[target_data.index.isin(train_index)]
                y_test = target_data[target_data.index.isin(test_index)]
            else:
                y_train, y_test = target_data[train_index], target_data[test_index]
            
            # Train models based on target type
            if target_name == "Domain":
                best_model_name, fold_results = classifier.train_domain_ml_models(
                    x_train, y_train, x_test, y_test
                )
            else:
                best_model_name, fold_results = classifier.train_multilabel_ml_models(
                    x_train, y_train, x_test, y_test
                )
            
            # Add fold number and best model to results
            fold_results["Fold"] = fold
            fold_results["Best_Model"] = best_model_name
            
            # Concatenate results
            result_df_fullFolds = pd.concat([result_df_fullFolds, fold_results], ignore_index=True)
        
        # Determine output directory and filename
        output_dir = base_path / ("DomainTaskResults" if target_name == "Domain" else "MLTaskResults")
        filename = f"{emb_name}_results.csv"
        output_path = output_dir / filename
        
        # Export results to CSV
        result_df_fullFolds.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # Optional: Calculate and save average metrics per model
        avg_metrics = result_df_fullFolds.groupby('Model').mean(numeric_only=True)
        avg_metrics_path = output_dir / f"{emb_name}_average_metrics.csv"
        avg_metrics.to_csv(avg_metrics_path)
        print(f"Average metrics saved to: {avg_metrics_path}")