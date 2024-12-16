
# **CanWeTrustReFAIR: A Replication Study** 🔍⚖️  

## **Overview** 📝  
This repository replicates and extends ReFAIR, a **Context-Aware Recommender for Fairness Requirements Engineering**. Our study evaluates the reproducibility of ReFAIR's findings and explores opportunities for improvement in fairness-aware systems.  

### Objectives:  
- **🔍 Understand** fairness challenges in requirements engineering.  
- **📊 Validate** the ReFAIR framework's performance on domain and multi-label classification tasks.  
- **🚀 Enhance** the framework for improved robustness, reliability, and impact.  

---

## **Project Structure** 📂  

- **Dataset/**: Contains all datasets used in testing and replication.  
- **Evaluation/**: Subfolders for results of RQ1 and RQ2.  
- **MLModels/**: Notebooks for training the best models in ReFAIR. Pre-trained weights are also stored here.
- **main.py**: Executes training and evaluation pipelines for RQ1 and RQ2.  
- **MoJo\_Distance.py**: Script for calculating results for RQ3.  
- **requirements.txt**: Contains all Python dependencies for the project.  

---

## **Installation Requirements** ⚙️  

### **1. Clone the Repository**  
```bash
git clone https://github.com/your-repo-link/CanWeTrustReFAIR.git
cd CanWeTrustReFAIR
```

### **2. Install Required Python Packages**  
Install dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

Alternatively, install packages manually:  
```bash
pip install numpy pandas scikit-learn transformers nltk gensim fasttext lazypredict
pip install scikit-multilearn xgboost
pip install torch  # Required for BERT
```

### **3. Pre-trained Embedding Models**  
The code will automatically download large pre-trained embedding models:  

| Model                 | Size       | Notes                          |
|-----------------------|------------|--------------------------------|
| **FastText**          | ~6.8 GB    | `cc.en.300.bin`                |
| **Word2Vec**          | ~3.5 GB    | `word2vec-google-news-300.bin` |
| **GloVe**             | ~800 MB    | Zipped and converted to Word2Vec format. |  

**Storage Requirement**: Ensure at least **12GB** of free disk space for embedding models and additional space for the Python environment and project files.

---

## **How to Run the Project** ▶️  

1. **Main Pipeline**:  
   Run the `main.py` script to execute training and evaluation for RQ1 (Domain Classification) and RQ2 (Multi-Label Classification):  
   ```bash
   python main.py
   ```  

2. **RQ3: MoJo Distance**:  
   After running `main.py`, execute `MoJo_Distance.py` to compute results for RQ3:  
   ```bash
   python MoJo_Distance.py
   ```  

3. **Evaluation Results**:  
   Results for RQ1 and RQ2 will be saved under the `Evaluation/` folder.  

---

## **Key Features** 🚀  

- **Reproducibility**: Fully replicates ReFAIR's domain and multi-label classification tasks.  
- **Flexible Pipeline**: Supports multiple word embeddings (TF-IDF, Word2Vec, GloVe, FastText, BERT).  
- **Multi-Label Classification**: Integrates Binary Relevance, Label Powerset, and Classifier Chains methods using scikit-multilearn.  
- **Evaluation Metrics**: Reports F1-Score and Hamming Loss for rigorous performance analysis.  

---

## **Results Overview** 📊  

| Research Question | Best Model                      | Embedding      | Metric          | Score      |
|-------------------|---------------------------------|----------------|-----------------|------------|
| **RQ1**           | XGBClassifier                  | BERT           | F1-Score        | 98.4\%     |
| **RQ2**           | LinearSVC + Label Powerset     | GloVe          | F1-Score        | 88.9\%     |
|                   |                                 |                | Hamming Loss    | 0.392      |

---

## **References** 📚  

1. **Ferrara, C., Casillo, F., Gravino, C., De Lucia, A., \& Palomba, F.** (2024, April).  
   *ReFAIR: Toward a Context-Aware Recommender for Fairness Requirements Engineering.*  
   In *Proceedings of the IEEE/ACM 46th International Conference on Software Engineering* (pp. 1–12).  

2. **Chen, Z., Zhang, J. M., Sarro, F., \& Harman, M.** (2022, November).  
   *MAAT: A Novel Ensemble Approach to Addressing Fairness and Performance Bugs for Machine Learning Software.*  
   In *Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering* (pp. 1122–1134).  

---

## **Contributors** 👥  

- **Ahmed Radwan**  
- **Claudia Farkas**  
- **Amir Heari**
