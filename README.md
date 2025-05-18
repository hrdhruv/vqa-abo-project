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

