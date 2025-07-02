# Authenticity‑Driven Disaster Tweet Classification (2025‑Spring DM)


* **Team Members**  
  - Lulin Yang   
  - Wendi Li  
  - Xiaoxuan Qin  
* **Project presentation**: 
  <a href="docs/presentation_slides.pdf" target="_blank" rel="noopener">
    🖥️ View Presentation Slides
  </a>

* **Project paper**: 
  <a href="docs/report_paper.pdf" target="_blank" rel="noopener">
    🖥️ View Report Paper
  </a>


## Description
This project detects and classifies disaster-related tweets on Twitter. We clean and preprocess raw tweet text, then extract features using Bag‑of‑Words, TF‑IDF without and with PCA, GloVe word embeddings, and BERT sentence embeddings. We train Logistic Regression, SVM, and Neural Network models, evaluate them with Accuracy, Precision, Recall, F1‑score, and AUC, and perform error analysis on false positives. Finally, we apply unsupervised clustering to distinguish subtypes of detected emergency tweets.


## Prerequisites
### System
- **R** ≥ 4.0  
- **Python** ≥ 3.7

### R packages

```r
install.packages(c(
  ## 1. Data Manipulation
  "tidyverse",    # dplyr, tidyr, etc. for data wrangling
  "data.table",   # super‐fast data frames
  
  ## 2. Text Preprocessing
  "tidytext",     # tokenization & text mining verbs
  "tm",           # classic corpus & DTM tools
  "SnowballC",    # word stemming
  "text2vec",     # vectorizers, TCM, word‐embedding training
  
  ## 3. Modeling & Tuning
  "caret",        # unified train/tune interface
  "glmnet",       # lasso/ridge/logistic regression
  "e1071",        # SVM (libsvm wrapper)
  "kernlab",      # alternative SVM backend
  "nnet",         # feed‑forward neural networks
 
  
  ## 4. Evaluation Metrics
  "pROC",         # ROC curves, AUC calculations
  
  ## 5. Visualization
  "ggplot2",      # core plotting library
  "ggpubr",       # publication‑ready ggplot helpers
  "wordcloud",    # word‐cloud generation
  "RColorBrewer", # qualitative & sequential palettes
  
  ## 6. Reporting & Utilities
  "knitr",        # R Markdown rendering
  
))
```
### Python Packages

```
pip install sentence-transformers numpy
```

## Data & Model Artifacts

To avoid bloating the repo, we host large precomputed assets on Google Drive. Please download the following files and place them in the corresponding folders: 

**Google Drive Link**: https://drive.google.com/drive/folders/1anm1pPPj4va1_eXv73enVHan13p2OUW3?dmr=1&ec=wgc-drive-globalnav-goto

| Data                                      | Destination                              | Google Drive Link                                 |
|-------------------------------------------|-------------------------------------------|---------------------------------------------------|
| Raw dataset (`train.csv`)                 | `data/train.csv`                          | [Download](https://drive.google.com/file/d/1T_JWDqKeH83HItuDhMj3ya_6P9Lsiq6g/view?usp=drive_link) |
| PCA components of Bag‑of‑Words            | `data/pca_bag_words.RData`                | [Download](https://drive.google.com/file/d/1H2HJ6teyEn6tlqS0lhxm6sXl3dP0AkgX/view?usp=drive_link) |
| PCA components of TF‑IDF                  | `data/pca_tfidf.RData`                    | [Download](https://drive.google.com/file/d/15YOAf3DzVtzBLI_NWs4LhfMI-jlVuj-3/view?usp=drive_link) |
| BERT sentence embeddings                  | `data/dtm_bert.npy`                       | [Download](https://drive.google.com/file/d/13yL63KPz0V86grqeeUIWL7Bf0faowXU2/view?usp=drive_link) |

And we have the following selective trained models, because of long time of training.

| Trained model                              | Destination                              | Google Drive Link                                 |
|--------------------------------------------|------------------------------------------|---------------------------------------------------|
| BoW Logistic regression                    | `models/bow_logit_model_0.rds`           | [Download](https://drive.google.com/file/d/1ffHjUpe-msWToJS5s6clqdM6BzbBZ6iV/view?usp=drive_link) |
| TF‑IDF Logistic regression                 | `models/tfidf_logit_model_0.rds`         | [Download](https://drive.google.com/file/d/1zorYJDzcbh48TmVdneyR7qeHFpABc_pX/view?usp=drive_link) |
| BoW SVM (linear)                           | `models/svm_model.rds`                   | [Download](https://drive.google.com/file/d/1bzIT1ZoZfoDM9zpQ92f9J47-K6fzCc7C/view?usp=drive_link) |
| TF‑IDF SVM (linear)                        | `models/svm_model_tfidf.rds`             | [Download](https://drive.google.com/file/d/1H3tWpZXoZRfuUrtEehyJwQY_MMJX7Lm0/view?usp=drive_link) |
| BoW + PCA SVM                              | `models/svm_model_pca.rds`               | [Download](https://drive.google.com/file/d/1g5ZNQIsBXTXDpAtjozyt3bem4KkUGyrp/view?usp=drive_link) |
| TF‑IDF + PCA SVM                           | `models/svm_model_pca_tfidf.rds`         | [Download](https://drive.google.com/file/d/1HmVkx4M0RX9A18cxS_xwkH3HB2ZANYXy/view?usp=drive_link) |
| GloVe SVM (radial)                         | `models/svm_model_embedding.rds`         | [Download](https://drive.google.com/file/d/1SGS356lDSmgyuP3Qz-mSuZJOvDScfuWB/view?usp=drive_link) |
| BERT SVM (radial)                          | `models/svm_model_bert.rds`              | [Download](https://drive.google.com/file/d/1OjdNhcMOjmnhktOEgBFWVBLs0ideW7VQ/view?usp=drive_link) |
| BoW + PCA Neural‑Net                       | `models/neural_network_bag_of_words.RData` | [Download](https://drive.google.com/file/d/1rndkwjtFJ_nFREumKBjXVQn2XAChik_u/view?usp=drive_link) |
| TF‑IDF + PCA Neural‑Net                    | `models/neural_network_tfidf.RData`      | [Download](https://drive.google.com/file/d/1Tq-Le2COeL3IXGfr10TXxs8d4chWYo72/view?usp=drive_link) |
| GloVe Neural‑Net                           | `models/neural_network_glove.RData`      | [Download](https://drive.google.com/file/d/1CnQDfsMsIMPq204Qhn9ntknp52EG956l/view?usp=drive_link) |
| BERT Neural‑Net                            | `models/neural_network_bert.RData`       | [Download](https://drive.google.com/file/d/1IjOC5Fqv7dDk5ASiys0TG4z9_vDUdKL1/view?usp=drive_link) |

After downloading, your directory should look like:

```
.
├── data/  
│   ├── train.csv                          # Raw dataset  
│   ├── pca_bag_words.RData                # BoW + PCA components  
│   ├── pca_tfidf.RData                    # TF‑IDF + PCA components  
│   └── dtm_bert.npy                       # BERT sentence embeddings  
│
├── models/  
│   ├── bow_logit_model_0.rds              # BoW + Logistic Regression  
│   ├── tfidf_logit_model_0.rds            # TF‑IDF + Logistic Regression  
│   ├── svm_model.rds                      # BoW + SVM (linear)  
│   ├── svm_model_tfidf.rds                # TF‑IDF + SVM (linear)  
│   ├── svm_model_pca.rds                  # BoW + PCA + SVM  
│   ├── svm_model_pca_tfidf.rds            # TF‑IDF + PCA + SVM  
│   ├── svm_model_embedding.rds            # GloVe + SVM (radial)  
│   ├── svm_model_bert.rds                 # BERT + SVM (radial)  
│   ├── neural_network_bag_of_words.RData  # BoW + PCA + Neural Net  
│   ├── neural_network_tfidf.RData         # TF‑IDF + PCA + Neural Net  
│   ├── neural_network_glove.RData         # GloVe + Neural Net  
│   └── neural_network_bert.RData          # BERT + Neural Net  
│
├── R/                                     # R scripts by stage (optional)  
│   ├── 01_preprocessing.R  
│   ├── 02_feature_engineering.R  
│   ├── 03_modeling.R  
│   └── 04_evaluation.R  
│
├── docs/                                  # Project documentation  
│   ├── presentation_slides.pdf            # Presentation slides (PDF)  
│   └── report_paper.pdf                   # Final project report paper (PDF)  
│
├── disaster_detection_source_code.html    # Project Source Code (html)
├── disaster_detection_source_code.rmd     # Project Source Code (rmd)
│
├── README.md                              # This file   
└── .gitignore  
```

## Authors
  - Lulin Yang (email: luy30@pitt.edu)   
  - Wendi Li (email: wel242@pitt.edu)  
  - Xiaoxuan Qin (email: xiq33@pitt.edu)  

## Acknowledgments
- We gratefully acknowledge **Kaggle** for hosting the “Natural Language Processing with Disaster Tweets” competition and providing the labeled dataset that forms the basis of this study  [oai_citation_attribution:0‡Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/overview/evaluation?utm_source=chatgpt.com).  
- We thank the **Stanford NLP Group** for releasing the pre‑trained GloVe word embeddings, which we leveraged for semantic text representation  [oai_citation_attribution:1‡Wikipedia](https://en.wikipedia.org/wiki/GloVe?utm_source=chatgpt.com).  



## License
[MIT](https://choosealicense.com/licenses/mit/)
