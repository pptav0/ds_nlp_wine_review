# Wine Review NLP Project

## Overview
The **Wine Review NLP Project** is an advanced natural language processing (NLP) system designed to analyze and process wine reviews. The project leverages state-of-the-art machine learning models, including transformer-based architectures, to extract insights, classify sentiments, and provide personalized wine recommendations. Additionally, it integrates the `nlp_helpers` module for reusable utilities such as text preprocessing, exploratory data analysis (EDA), and visualization.

## Features
- **Flavor Tagging**: Classifies wine reviews into flavor profiles using a pre-trained transformer model.
- **Sentiment Analysis**: Converts wine ratings into sentiment categories (positive, neutral, negative).
- **Recommendation Engine**: Suggests wines based on user queries, combining strict and exploratory recommendations.
- **Efficient Search**: Uses FAISS for fast similarity search on encoded wine reviews.
- **Custom Training**: Implements a custom `Trainer` class for weighted binary classification tasks.
- **Reusable Utilities**: Leverages the `nlp_helpers` module for text preprocessing, EDA, and visualization.
- **Taxonomy Prediction**: Predicts wine variety and country using auxiliary classifiers.

---

## Folder Structure
```plaintext
wine_review/
├─ notebook/                         # Notebooks only (no libs)
├─ tests/                            # Unit tests (under development)
├─ wine_review/                      # Source package
│  ├─ config.py                      # Paths, seeds, constants
│  ├─ modeling/                      # ML models (training/inference)
│  │  ├─ __init__.py
│  │  ├─ utils.py                    # WeightedBCETrainer, batching utils
│  │  ├─ flavors.py                  # Multi-label flavor classifier + infer_probs
│  │  ├─ sentiment.py                # Sentiment classifier + predict_sentiment
│  │  └─ taxonomy.py                 # Variety/Country classifiers (train/load/predict)
│  ├─ retrieval/                     # Semantic search + business rules
│  │  ├─ __init__.py
│  │  ├─ embed.py                    # build_index, encode_query, retrieve_candidates
│  │  └─ recommend.py                # apply_business_rules, optional LLM rerank hook
│  ├─ models/                        # (optional) exported ONNX, quantized or local weights
│  └─ reports/                       # (optional) generated HTML/MD reports
│
├─ LICENSE
├─ README.md
└─ pyproject.toml

nlp_helpers/                      # Shared, general-purpose NLP utilities
├─ __init__.py
├─ clean_data.py
├─ plots_eda.py
├─ emotion_analysis.py
├─ topic_modeling.py
└─ common_imports.py
```
