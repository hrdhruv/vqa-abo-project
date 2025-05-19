# VR_Mini_Project_2
# Multimodal Visual Question Answering with Amazon Berkeley Objects Dataset

## Introduction

This project focuses on building a Visual Question Answering (VQA) pipeline using multimodal datasets that combine images and metadata. The goal is to generate and evaluate one-word answer QnA pairs using state-of-the-art models such as **BLIP-VQA-Base**, **BLIP2**, and **ViLBERT**. A custom dataset was curated using the **Gemini API**, and fine-tuning was performed using **LoRA**. The models were evaluated using accuracy and semantic metrics to optimize performance for real-world product understanding tasks.


## Overview
This project involves creating a **Visual Question Answering (VQA)** system using the Amazon Berkeley Objects (ABO) dataset. The pipeline includes:
- Generating a multiple-choice VQA dataset using multimodal APIs or on-device models.
- Evaluating baseline performance using pre-trained models.
- Fine-tuning selected models with **LoRA (Low-Rank Adaptation)**.
- Evaluating model performance using standard and proposed metrics.
- Deploying an inference script for the final trained model.

---

## Dataset
**Amazon Berkeley Objects (ABO)**  
- 147,702 product listings with multilingual metadata  
- 398,212 unique catalog images  
- *Use the small variant (3GB, 256x256 images + CSV metadata)*

**Download Link:** [ABO Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)

---


## 📁 Repository Structure

| File/Folder                         | Description |
|------------------------------------|-------------|
| `data-curation-kaggle/colab.ipynb` | Notebook for data curation using the Gemini 2 API to generate questions and answers from product metadata and images. |
| `blip-vqa-80.ipynb`                | Fine-tuning code for the `blip-vqa-base` model on the curated Q&A dataset. |
| `blip2-finetune.ipynb`             | Fine-tuning code for the `Blip2-Flan-T5-XL` model on the curated Q&A dataset. |
| `vilbert-fine-tune.ipynb`          | Fine-tuning code for the `ViLBERT` model on the curated Q&A dataset. |
| `vilbert_baseline.ipynb`           | Evaluation code for the off-the-shelf `ViLBERT` model (baseline without fine-tuning). |
| `evaluation.ipynb`                 | Comprehensive evaluation of all models on various dataset sizes using multiple metrics. |
| `cleaned_qna.csv` & `test.csv`     | Curated Q&A datasets used for model training and testing. |
| `inference_main/`                  | Contains the inference script required for final evaluation as per project specifications. |
| `kaggleinferencevalidation.zip`    | Includes Kaggle-based inference validation to demonstrate portability without requiring a Conda environment. |

---

## Pipeline

### 1. Data Curation

- Multimodal tools such as **Gemini 2.0 API** and on-device models via **Ollama** were explored. Ultimately, **Gemini 2.0 API** was selected for its superior performance in generating relevant Q&A pairs.
- For each image, **3–7 question-answer pairs** were generated.
- Curated **single-word answer** Visual Question Answering (VQA) samples.  
  *Example:*  
  **Q:** What is the color of the bag?  
  **A:** Red
- Emphasis was placed on **diversity in question types** and **varying difficulty levels**, particularly by leveraging product metadata.
- **Duplicate rows** were removed. Among duplicates, rows with the most complete metadata were retained—language was not a constraint, as Gemini could convert content into English effectively.
- Careful **prompt engineering** was employed to:
  - Encourage one-word answers
  - Minimize bias toward “True/False” or purely numerical answers
  - Ensure metadata was considered alongside image context for meaningful question generation


### 2. Baseline Evaluation

- Baseline evaluation was performed using the following pre-trained models:
  - **BLIP-2**
  - **BLIP-VQA-BASE**
  - **ViLBERT**
- These models were evaluated **without any fine-tuning** to establish reference performance.
- Metrics recorded include:
  - **Accuracy** and **F1 Score**
  - **BERTScore** for semantic similarity
  - **Ontological similarity** and **Cosine similarity** (useful for one-word answer comparison)
- Baseline inference was run on subsets of the curated test set:
  - Evaluations were done on **100** and **1000** test samples as available in `test.csv` using `evaluation.py`.

### 3. Fine-Tuning with LoRA

- Fine-tuning was performed using **LoRA (Low-Rank Adaptation)** in combination with **PEFT (Parameter-Efficient Fine-Tuning)** and **quantization** techniques to reduce resource requirements.
- Due to **GPU memory** and **compute constraints**, models were trained on smaller subsets of the dataset with fewer epochs.
- An **iterative training approach** was used:
  - Initially trained on a smaller portion of the dataset.
  - Additional samples were used progressively, especially for **BLIP-2**, where memory limitations restricted batch size and sequence length.
- Various **LoRA hyperparameters** were experimented with, including:
  - `r` (rank)
  - `alpha` (scaling factor)
  - `target_modules` (specific layers for LoRA injection)
  - `task_type` (e.g., question answering)
- The best-performing configurations were selected based on a combination of:
  - **Accuracy**, **F1 Score**
  - **BERTScore**
  - **Cosine similarity** and **ontological similarity**
- This fine-tuning allowed better adaptation to the curated QnA data, especially for one-word answers grounded in both **image content** and **metadata context**.

### 4. Evaluation Metrics

To measure the performance of models on single-word Visual Question Answering (VQA), we employed both traditional and semantic evaluation metrics.

- **Primary Metrics:**  
  - **Accuracy**: Measures exact match between predicted and ground-truth answers.  
  - **F1 Score**: Useful for imbalanced answer distributions, especially with sparse label spaces.  

- **Additional Metrics:**  
  - **BERTScore** and **BARTScore**: Evaluate the semantic similarity between predicted and actual answers, beneficial for short text like single-word outputs.  
  - **Ontological Similarity**: Measures how semantically close two words are within a knowledge graph (e.g., WordNet).  
  - **Cosine Similarity**: Vector-based semantic similarity between embeddings of predicted and ground truth answers.

---

#### Performance Comparison: Before vs After Fine-Tuning (LoRA)

| Model              | Stage             | Accuracy | F1 Score | BERTScore | Cosine Sim. |
|-------------------|-------------------|----------|----------|------------|--------------|
| **BLIP-VQA-Base**  | Before Fine-Tuning| 34%      | 0.16     | 0.850      | 0.633        |
|                    | After LoRA FT     | 54%      | 0.31     | 0.960      | 0.736        |
| **BLIP2-Flan-T5**  | Before Fine-Tuning| 20%      | 0.09     | 0.779      | 0.490        |
|                    | After LoRA FT     | 49%      | 0.275    | 0.870      | 0.721        |
| **ViLBERT**        | Before Fine-Tuning| 18%      | 0.06     | 0.855      | 0.510        |
|                    | After LoRA FT     | 51%      | 0.298    | 0.910      | 0.728        |

---

#### Key Observations:

- **BLIP-VQA-Base** emerged as the most balanced and effective model post fine-tuning. It achieved the **highest accuracy and semantic similarity scores**, making it the preferred choice for the final inference stage.
  
- **BLIP2-Flan-T5** is a **large generative model**, originally designed to produce complete sentences. While it’s powerful, its tendency to generate verbose responses (e.g., full phrases instead of single-word answers) led to **lower accuracy and F1**, despite moderate semantic similarity scores. It improved with prompt engineering and LoRA-based tuning.

- **ViLBERT**, though older, demonstrated strong label-focused prediction behavior post fine-tuning. Its performance improved significantly when fine-tuned, especially in BERTScore and cosine similarity, indicating its capability to learn concise and context-aware responses.

> 🔍 **Note:** Fine-tuning was done using **LoRA**, **PEFT**, and **quantization**, and was adapted to memory-constrained environments by iteratively training on smaller curated datasets with single-word answers.

### 5. Iterative Improvement

- **Continuous refinement** of both the dataset and fine-tuning strategy was done based on feedback from evaluation metrics (Accuracy, BERTScore, Cosine similarity, etc.).
- Prompt engineering and metadata usage were revised to improve the **quality and diversity of generated QnA pairs**.
- Fine-tuning parameters in LoRA were **adjusted across iterations**, using smaller data batches to monitor incremental gains without overfitting.
- Observations from **baseline vs. fine-tuned model performance** guided decisions on data augmentation, hyperparameter tuning, and layer targeting.
- The goal was to **maximize performance on evaluation metrics** and ensure generalizability before final submission and deployment.

--- 

## Future Improvements  

- **Joint Multimodal Curation**: Currently, question-answer pairs are primarily generated from metadata. Future work should emphasize joint use of both **image features and metadata** for more diverse and visually grounded questions.

- **Larger-Scale Training**: Fine-tuning was done with limited data and compute. With access to **larger datasets** and **longer training schedules**, especially with LoRA or full fine-tuning, performance can be significantly enhanced.

- **Better Prompt Engineering for Generative Models**: Especially for BLIP2, improved prompt design can guide the model toward **more concise, one-word outputs** instead of full sentences.

- **Hard Negative Sampling**: Incorporating harder or semantically similar distractors in the training set can help models **better distinguish subtle visual cues**.

- **Evaluation with Human Judgement**: Complement automated metrics with **human evaluation** to assess answer appropriateness, especially for ambiguous or subjective questions.

- **Ensemble of Models**: Combine outputs from multiple models (e.g., BLIP2 + BLIP-VQA) to leverage **both generative and classification-style answering**.

- **Task-Specific Adapters**: Explore **vision-language adapters or task-specific heads** that specialize in single-word classification from image-question pairs.

- **Error Analysis & Bias Mitigation**: Conduct systematic error analysis to identify common failure patterns (e.g., color confusion, language bias) and improve robustness.

---

## Member-wise Contribution  

| Member           | GitHub Handle | Contribution |
|------------------|---------------|--------------|
| **Harsh Dhruv**  | [@hrdhruv](https://github.com/hrdhruv) | Developed the **BLIP-VQA-Base** baseline and fine-tuned model, including implementation of the inference pipeline. |
| **Rudra Pathak** | [@rudra0000](https://github.com/rudra0000) | Worked on the **ViLBERT** baseline and fine-tuning, along with model evaluation and reporting. |
| **Aditya Saraf** | [@NikaYz](https://github.com/NikaYz) | Implemented the **BLIP-2** baseline and fine-tuned model, and contributed to project documentation and the README. |

> **Note**: All three members contributed equally to the **data curation** process, including prompt engineering, metadata extraction, and question-answer generation.

---
<!-- 
# vqa-abo-project
AIM825 Course Project: Multimodal Visual Question Answering with Amazon Berkeley Objects Dataset
Overview
This repository contains the code, dataset, and documentation for the AIM825 Course Project at IIITB, developed by Harsh Dhruv (IMT2022008), Rudra Pathak (IMT2022081), and Aditya Saraf (IMT2022067). The project focuses on building a Multimodal Visual Question Answering (VQA) system using the Amazon Berkeley Objects (ABO) dataset. We curated a dataset of approximately 100,000 single-word answer question-answer pairs, evaluated pre-trained models, fine-tuned BLIP-VQA-Base, BLIP-2 (Flan-T5-XL), and Vilbert using Low-Rank Adaptation (LoRA), and assessed performance with metrics like Exact Match Accuracy, F1 Score, BERTScore, BARTScore, Cosine Similarity, and Ontological Similarity. The project was constrained to free cloud GPUs (Kaggle) and models with up to 7 billion parameters.
Project Components
1. Data Curation

Dataset: Created a VQA dataset with ~100,000 question-answer pairs from the ABO dataset (small variant, 3GB), saved as cleaned_qna.csv.
Process: Used Gemini 2.0 API to generate 3–7 single-word answer questions per image, incorporating metadata (color, material, item name). Merged image and metadata CSVs, validated paths, and visualized outputs.
Challenges: Managed API rate limits, ensured valid image paths, and improved question diversity.

2. Baseline Evaluation

Models: Evaluated pre-trained BLIP-VQA-Base on an 8,000-sample validation set.
Metrics: Computed Exact Match Accuracy (34.00%), F1 Score (16.64%), BERTScore (F1: 85.23%), Cosine Similarity (0.6335), and Ontological Similarity (0.7236) for BLIP-VQA-Base, with similar metrics for BLIP-2 and Vilbert.

3. Fine-Tuning with LoRA

BLIP-VQA-Base: Fine-tuned on 80,000 samples using LoRA (rank=32, alpha=64) with mixed precision training and a cosine scheduler.
BLIP-2 (Flan-T5-XL): Fine-tuned on smaller samples (~900) with 8-bit quantization, LoRA (rank=128, alpha=128), and a prompt template ("Answer in one word only").
Vilbert: Fine-tuned on 80,000 samples, aligning Faster R-CNN image features with text embeddings, using LoRA (rank=64, alpha=32) and a linear scheduler.
Challenges:
BLIP-VQA-Base: Memory constraints and tokenization for single-word answers.
BLIP-2: NaN losses, gradient instability, and full-sentence generation.
Vilbert: Complex image feature processing, modality alignment, and training stability.



4. Inference

Script: Located in the inference_main folder, loads fine-tuned models with LoRA adapters, processes ABO images and questions from a metadata CSV, and generates single-word answers using beam search.
Output: Saves predictions to results.csv with ground-truth and generated answers, evaluated using BERTScore.

5. Results

Fine-Tuned Performance:
BLIP-VQA-Base: Accuracy: 54.00%, F1: 31.44%, BERTScore F1: 96.23% (validation), 87% (test).
BLIP-2: Accuracy: 49.00%, F1: 27.58%, BERTScore F1: 77.96%.
Vilbert: Accuracy: 51.00%, F1: 29.88%, BERTScore F1: 91.89%.


Insights: BLIP-VQA-Base outperformed others due to its ability to generate concise answers and larger training dataset. All models improved over baselines, leveraging LoRA and optimization techniques (mixed precision, quantization, gradient clipping).

Repository Structure

cleaned_qna.csv: Curated dataset with ~100,000 question-answer pairs.
inference_main/: Contains the inference script (inference.py) and dependencies.
report/: Project report PDF and LaTeX source.
notebooks/: Jupyter notebooks for data curation, fine-tuning, and evaluation (if applicable).
requirements.txt: Python dependencies for the project.

Setup and Running Inference

Clone the Repository:
git clone https://github.com/hrdhruv/vqa-abo-project.git
cd vqa-abo-project


Install Dependencies:

Create a Conda environment:conda create -n vr-eval python=3.8
conda activate vr-eval


Install requirements:pip install -r requirements.txt




Prepare Data:

Place the ABO dataset images in a directory (e.g., /path/to/abo-images).
Ensure cleaned_qna.csv or a metadata CSV with image paths and questions is available.


Run Inference:
python inference_main/inference.py --image_dir /path/to/abo-images --csv_path cleaned_qna.csv


Output: results.csv with ground-truth and generated answers.


Evaluate Results:

Use BERTScore to evaluate results.csv (script included in inference_main/evaluate.py, if provided).



Deliverables

GitHub Repository: https://github.com/hrdhruv/vqa-abo-project
Dataset: cleaned_qna.csv (~100,000 question-answer pairs)
Inference Script: In inference_main folder
Report: Detailed project report in report/ folder

Dependencies

Python 3.8+
PyTorch, Transformers, Pandas, NumPy, PIL, BERTScore, BARTScore
Kaggle GPU environment for fine-tuning and inference
See requirements.txt for the full list
-->
