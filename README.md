# 🐞 BugClassiNet

**DeepBugSIM** is a deep learning-based tool that automatically classifies software bug reports as real bugs or non-bugs (e.g., feature requests, duplicates, invalid entries). It uses semantic embeddings from a fine-tuned Sentence-BERT model and feeds them into a custom 1D Convolutional Neural Network (CNN) for accurate classification.

---

## 🚀 Features

- ⚙️ Parses XML bug reports and extracts relevant text fields  
- 🧠 Generates semantic embeddings using Sentence-BERT  
- 🧪 Classifies reports using a lightweight CNN model  
- 📊 Handles class imbalance via synthetic data generation  
- 📁 Easy to extend, test, and integrate into larger systems

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/DeepBugSIM.git
cd DeepBugSIM
pip install -r requirements.txt
