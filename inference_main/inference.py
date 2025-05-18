import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import gdown
import os
from PIL import Image
from transformers import logging
from peft import PeftModel
from transformers import BlipProcessor, BlipForQuestionAnswering
import warnings
warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint")

# Suppress logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
logging.set_verbosity_error()

def download_weights():
    output_path = "/kaggle/working/lora_weights"
    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(os.path.join(output_path, "adapter_model.bin")) or \
       not os.path.exists(os.path.join(output_path, "adapter_config.json")):
        # https://drive.google.com/drive/folders/16AEaDufitpjYBU7mZ0D24NnL_SlcCvZc/
        gdown.download_folder("https://drive.google.com/drive/folders/1jQEE8JnISg91hCS-XZTEpg-cftYZNT0O", output=output_path, quiet=False)

    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    args = parser.parse_args()
    print("model")
    df = pd.read_csv(args.csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    base_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

    finetuned_path = download_weights()

    model = PeftModel.from_pretrained(base_model, finetuned_path)

    model.eval()
 
    generated_answers = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.image_dir, row["image_name"])
        question = str(row["question"])

        try:
            image = Image.open(image_path).convert("RGB")
            prompt = question
            encoding = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**encoding, max_new_tokens=4)
            answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip().lower()
            answer = answer.split()[-1]
        except Exception as e:
            print(f"Error at {image_path}: {e}")
            answer = "error"

        generated_answers.append(answer)
    print(question)

    print(generated_answers)
    df["generated_answer"] = generated_answers
    print(df.head())
    df.to_csv("/kaggle/working/results/results.csv", index=False)

if __name__ == "__main__":
    main()
