# **CanWeTrustReFAIR: A Replication Study** 🔍⚖️

## **Overview** 📝  
This repository is dedicated to replicating and building upon ReFAIR, a Context-Aware Recommender for Fairness Requirements Engineering. Our goal is to explore fairness in requirements engineering by reproducing the findings of the original work and identifying opportunities for further improvement.

Through this replication, we aim to:
- **🔍 Understand** fairness challenges in requirements engineering.  
- **📊 Evaluate** the ReFAIR framework and validate its results.  
- **🚀 Enhance** the framework to make fairness-aware recommendations more reliable and impactful.  

---
## Installation Requirements

### Python Packages
```bash
pip install numpy pandas scikit-learn transformers nltk gensim fasttext lazypredict
pip install torch  # Required for BERT
```

### Required Models
The code will automatically download several large pre-trained embedding models:

1. FastText Model (cc.en.300.bin)
   - Size: ~6.8 GB
   - Downloads automatically

2. Word2Vec Model (word2vec-google-news-300.bin)
   - Size: ~3.5 GB
   - Downloads automatically

3. GloVe Embeddings
   - Size: ~800MB (zipped)
   - Downloads automatically
   - Converts to Word2Vec format

### Storage Requirements
- At least 12GB of free disk space for embedding models
- Additional space for Python environment and project files

### Dataset
Required files:
- Multiple Datasets in the Folder

---
## References 📚
1. **Chen, Z., Zhang, J. M., Sarro, F., & Harman, M.** (2022, November).  
   *MAAT: A Novel Ensemble Approach to Addressing Fairness and Performance Bugs for Machine Learning Software.*  
   In *Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering* (pp. 1122–1134).  

2. **Ferrara, C., Casillo, F., Gravino, C., De Lucia, A., & Palomba, F.** (2024, April).  
   *ReFAIR: Toward a Context-Aware Recommender for Fairness Requirements Engineering.*  
   In *Proceedings of the IEEE/ACM 46th International Conference on Software Engineering* (pp. 1–12).  
