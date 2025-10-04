# Wine Review NLP Project

## Overview
The **Wine Review NLP Project** is an advanced natural language processing (NLP) system designed to analyze and process wine reviews.
It leverages state-of-the-art transformer models to extract insights, classify sentiments, and predict wine characteristics such as flavor, variety, and country.
This work forms the foundation for a future **AI-driven recommendation engine**, capable of suggesting wines based on user tasting preferences expressed in natural language.

Developed as part of the **Cambridge Data Science Program**, the project demonstrates scalable transformer-based modeling with high accuracy and reusable components for experimentation and deployment.

## Features
- **Flavor Tagging**: Classifies wine reviews into flavor profiles using a fine-tuned transformer model.
- **Sentiment Analysis**: Converts wine ratings into sentiment categories (positive, neutral, negative).
- **Taxonomy Prediction**: Predicts wine **variety** and **country** using auxiliary classifiers.
- **Recommendation Engine**: Suggests wines by combining strict and exploratory matches based on user prompts.
- **Semantic Retrieval (FAISS-free)**: Uses `sentence_transformers.util.semantic_search` to retrieve semantically similar reviews — eliminating FAISS dependency for simpler, scalable execution.
- **Custom Training**: Implements a custom `WeightedBCETrainer` for handling class imbalance in multi-label tasks.
- **Reusable Utilities**: Integrates the `nlp_helpers` package for text preprocessing, exploratory data analysis (EDA), and visualization.

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
