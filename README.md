# 📖 SeeKRec: Toward Semantic-empowered Knowledge-aware Recommendation

## 📂 Dataset

We use the same datasets as in **CIKG Rec**:  

- [📄 Paper Link](https://arxiv.org/abs/2412.13544)  
- [💻 GitHub Link](https://github.com/laowangzi/CIKGRec)  

## ⚙️ Environment Requirement
Our experiments are conducted using **Python 3.8**. The required packages are:

- torch == 2.1.0  
- pandas == 2.0.3  
- scikit-learn == 1.3.2  
- numpy == 1.24.4  

👉 For convenience, you may install them via:
```bash
pip install -r requirements.txt
```

📦 Additionally, please create a directory named `pretrained_models` and download the pretrained text-embedding model **sup-simcse-roberta-large** from Hugging Face:
 [🔗 Download Link](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large)

## 🚀 Quick Start
To run the code (take dbbook2014 dataset as an example):

```
cd models
python main.py --dataset dbbook2014
```

## 📑Details of Important Files
* **`User_Preference_Extraction_Input.py`**: Prepares input for the LLM to extract users’ semantic preferences.
* **`User_Preference_Extraction_Output.py`**: Processes and formats the LLM’s output of users’ semantic preferences.
* **`Preference_Align_Input.py`**:Constructs input for the LLM to align semantically equivalent preferences.
* **`Preference_Align_Output.py`**: Retrieves and organizes the LLM’s aligned preference outputs.
* **`Structure_User_Semantics.py`**: Further refine user semantic alignment (optional) and acquire structured user semantic knowledge.
* **`Subgraph_Sampling.py`**: Samples semantic subgraphs of users and items using the proposed Search–Prune–Evaluate strategy.
* **`Enhance_Subgraph_Input.py`**: Generates input for the LLM to enrich subgraph semantics.
* **`Enhance_Subgraph_Output.py`**: Captures the LLM’s enhanced semantic representations of users and items.
* **`Get_Semantic_Embedding.py`**:Converts the LLM-enhanced semantics into continuous embedding vectors.

