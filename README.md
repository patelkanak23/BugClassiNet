# 🐞 BugClassiNet

**BugClassiNet** is a deep learning-based framework for automatically classifying software bug reports into **real bugs** or **non-bugs** (such as feature requests, duplicates, or invalid entries). It leverages the power of **Sentence-BERT** to extract rich semantic embeddings and classifies these using a custom-designed **1D Convolutional Neural Network (CNN)** for high accuracy.

---

## 🚀 Features

- ⚙️ **XML Bug Parsing** — Extracts textual information (e.g., title, description) from structured XML bug reports  
- 🧠 **Semantic Embeddings** — Uses a fine-tuned **Sentence-BERT** model to encode reports into contextual vector representations  
- 🧪 **Deep Classification Model** — Applies a lightweight and efficient **1D CNN** for classification  
- 🧬 **Data Augmentation** — Handles class imbalance through synthetic data generation techniques (e.g., SMOTE or controlled duplication)  
- 🔌 **Modular Design** — Easy to plug into existing bug tracking pipelines or issue management systems  
- 📈 **Performance-Ready** — Designed with scalability, fast inference, and extensibility in mind

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/DeepBugSIM.git
cd DeepBugSIM
pip install -r requirements.txt
