# Multi-Modal-Meme-Virality-Prediction-using-HGNNs

This repository showcases a mini-project developed for **DLG UE22AM342BA2: Deep Learning on Graphs**. It presents a multimodal meme virality predictor that integrates image, text, and metadata through ensemble learning and Hypergraph Neural Networks to classify memes as viral or non-viral.

---

## Project Structure
```
├── AlternativeApproach_DLG_078_079_089.ipynb # HGNN-based approach notebook
├── MainApproach_DLG_078_079_089.ipynb # Multimodal ensemble approach notebook
├── OCR Notebook.ipynb # Text extraction and preprocessing via OCR
├── GDL_078_079_089_PPT.pptx # Presentation slides for the project
├── Output # Folder containing files generated
└── README.md # Project documentation
```

## Dataset

We use a Reddit-based meme dataset with over 5,600 memes. It includes:
- Images from various meme subreddits
- Extracted text from images using OCR
- Metadata: timestamps, subreddit name, post score

**Download**: [Kaggle - Meme Dataset](https://www.kaggle.com/datasets/musadiqpashak/meme-dataset)

---

## Problem Statement

Virality on social media is influenced by diverse factors: visual cues, textual content, timing, and platform dynamics. This project predicts whether a meme will go viral or not using multimodal inputs and innovative learning techniques.

---

## Approaches

### Approach 1: Multimodal Ensemble Model

#### Overview
This approach independently processes each data modality (image, text, and metadata) through specialized models and then combines their outputs using late fusion (ensemble averaging).

#### Components

1. **Textual Pipeline (OCR + BERT)**
   - OCR: Used PyTesseract to extract text from meme images.
   - Embedding: Tokenized and embedded extracted text using pre-trained BERT.
   - Classification: Feed-forward neural network trained on BERT embeddings.
   - Explainability: Attention visualization and SHAP were used to interpret token-level importance.

2. **Visual Pipeline (CLIP-based Embeddings)**
   - Model: Used OpenAI’s CLIP to embed meme images into a 512-dimensional latent space.
   - Classifier: Small MLP trained on these embeddings.

3. **Tabular Metadata Pipeline**
   - Features used: Time of post, Subreddit, Score, Post hour
   - Processing: One-hot encoding, binning, and SHAP analysis.

4. **Ensemble Fusion**
   - Combined predictions from all three classifiers using weighted averaging.

#### Results
- Accuracy: ~69%
- Highlights: Best performance with balanced precision and recall.

---

### Approach 2: Hypergraph Neural Network (HGNN)

#### Overview
Constructs a hypergraph where each node represents a meme and hyperedges represent shared metadata features (e.g., time bins, subreddit clusters). Node features are created by concatenating CLIP and BERT embeddings.

#### Methodology

1. **Feature Fusion**
   - Node vector = [CLIP (512-d) || BERT (768-d)] = 1,280-d vector

2. **Hyperedge Construction**
   - Based on similar:
     - Subreddits
     - Time bins
     - Score clusters

3. **Learning**
   - Used a Hypergraph Convolution Layer (HyperConv)
   - Final softmax layer for classification

#### Results
- Accuracy: ~66–67%
- Highlights: Better recall on viral memes and good generalization through group-level structure

---

## Summary: Approach Comparison

| Metric / Aspect         | Approach 1: Ensemble           | Approach 2: Hypergraph GNN        |
|-------------------------|--------------------------------|------------------------------------|
| Modal Separation        | Handled independently          | Fused before learning              |
| Feature Fusion          | Late Fusion (output level)     | Early Fusion (embedding level)    |
| Metadata Usage          | As separate tabular classifier | For hyperedge formation           |
| Interpretability        | SHAP + Attention               | Cluster-level insights             |
| Best Metric             | Accuracy (~69%)                | Recall on viral memes              |
| Complexity              | Modular but separate pipelines | Unified but graph-heavy            |
| Scalability             | Easier to scale                | Complex edge creation logic        |

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score (macro and weighted average)

| Model           | Accuracy | Notes                          |
|----------------|----------|-------------------------------|
| Text (BERT)     | 0.62     | Weakest but interpretable     |
| Image (CLIP)    | 0.68     | Stable performance             |
| Hypergraph      | 0.67     | High recall for viral memes   |
| Ensemble        | 0.69     | Best overall performance      |


---

## Screenshots

#### Approach 1: Multimodal Ensemble Architecture
<img src="https://github.com/MusadiqPasha/Multi-Modal-Meme-Virality-Prediction-using-HGNNs/blob/main/Output/approach1.png" width="400" alt="Approach 1 Overview"/>

#### Approach 2: Hypergraph Neural Network Architecture
<img src="https://github.com/MusadiqPasha/Multi-Modal-Meme-Virality-Prediction-using-HGNNs/blob/main/Output/approach2.png" width="400" alt="Approach 2 Overview"/>

#### Meme Hypergraph Visualization
<img src="https://github.com/MusadiqPasha/Multi-Modal-Meme-Virality-Prediction-using-HGNNs/blob/main/Output/graph.png" width="400" alt="Graph Visualization"/>

---
## Challenges

- Quantifying virality consistently
- Fusing multimodal data effectively
- Building and validating the hypergraph structure

---

## References

- OpenAI CLIP: https://github.com/openai/CLIP
- BERT: https://huggingface.co/bert-base-uncased
- Hypergraph Neural Networks: https://arxiv.org/abs/1901.08150


