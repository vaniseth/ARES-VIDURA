import warnings
warnings.filterwarnings('ignore')

import os
import torch
import transformers
import re, random
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import PyPDF2
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel

# --- Load Environment Variables ---
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY_HOME")
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Gemini API Configuration ---
genai.configure(api_key=google_api_key)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')

# --- Model and Tokenizer Configuration ---
base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- PDF Data Source ---
pdf_folder_path = "DataSets/"

# --- Output Naming ---
# Stage 1 (Trained on Gemini data)
dataset_v1_path = "dataset_gemini_v1.json"
model_v1_output_dir = "cnt-finetune-v1-on-gemini-data"

# Stage 2 (Trained on Model v1's data)
dataset_v2_path = "dataset_mistral_v2.json"
model_v2_output_dir = "./cnt-finetune-v2-on-mistral-data"

# --- STEP 2: LOAD AND PROCESS ALL PDFS FROM A FOLDER ---
print("\n--- STEP 2: LOAD AND PROCESS ALL PDFS FROM A FOLDER ---")

def pdf_to_text_from_folder(folder_path, skip_start_pages=5, skip_last_pages=0, header_lines=2, footer_lines=1):
    combined_text = ""
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at {folder_path}")
        return ""
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}...")
            try:
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(pdf_reader.pages)
                    start_page = max(0, skip_start_pages -1)
                    end_page = num_pages - skip_last_pages

                    for page_num in range(start_page, end_page):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if not page_text: continue
                        
                        lines = page_text.splitlines(True)
                        if len(lines) > (header_lines + footer_lines):
                            lines = lines[header_lines:-footer_lines]
                        else:
                            lines = []
                        
                        combined_text += "".join(lines)
            except Exception as e:
                print(f"  Could not process {filename}. Error: {e}")
    return combined_text

raw_text = pdf_to_text_from_folder(pdf_folder_path)
print(f"\nTotal characters extracted from all PDFs: {len(raw_text)}")

# --- HELPER FUNCTIONS for Data Generation and Training ---
# (Defined here to be used in multiple stages)

def generate_response_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text if response.parts else None
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def generate_dataset(generation_function, num_samples, output_filename, **kwargs):
    if os.path.exists(output_filename):
        print(f"Dataset '{output_filename}' already exists. Skipping generation.")
        with open(output_filename, "r") as f:
            return json.load(f)

    data = []
    sample_char_lens = [1024, 512, 256]
    iterations_per_len = num_samples // len(sample_char_lens)

    for char_len in sample_char_lens:
        for i in range(iterations_per_len):
            print(f"Generating sample {len(data) + 1}/{num_samples} (length: {char_len})...")
            
            # Create a clean text sample
            random_start = random.randint(0, len(raw_text) - char_len)
            doc_sample = " ".join(raw_text[random_start:random_start + char_len].split()).replace("\u2010", "-")

            # Generate Q&A pair using the provided function
            qa_pair = generation_function(doc_sample, **kwargs)
            if qa_pair:
                data.append(qa_pair)

    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Successfully generated {len(data)} samples and saved to '{output_filename}'.")
    return data

def train_model(train_dataset, eval_dataset, base_model_id, output_dir, max_steps=100):
    print(f"\n--- Starting Training for model: {output_dir} ---")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", add_eos_token=True, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize data
    def tokenize_function(sample):
        user_content = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}"
        assistant_content = f"### Response:\n{sample['output']}"
        text = f"<s>[INST] {user_content.strip()} [/INST] {assistant_content.strip()} </s>"
        
        result = tokenizer(text, truncation=True, max_length=2048, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train = train_dataset.map(tokenize_function, remove_columns=list(train_dataset.features))
    tokenized_eval = eval_dataset.map(tokenize_function, remove_columns=list(eval_dataset.features))

    # Prepare model for QLoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_config)

    # Trainer setup
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        args=transformers.TrainingArguments(
            output_dir=output_dir, warmup_steps=5, per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=max_steps, learning_rate=2.5e-5, logging_steps=10, bf16=True, optim="paged_adamw_8bit",
            save_strategy="steps", save_steps=max_steps//2, eval_steps=max_steps//2,
            report_to="none"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    
    # Train
    trainer.train()
    
    # Save final model
    final_path = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_path)
    print(f"--- Finished Training. Final model saved to: {final_path} ---")
    return final_path

def prepare_and_save_splits(dataset_list, prefix):
    random.shuffle(dataset_list)
    train_size = int(0.9 * len(dataset_list)) # Use more data for training
    train_data = Dataset.from_list(dataset_list[:train_size])
    eval_data = Dataset.from_list(dataset_list[train_size:])
    print(f"Split for '{prefix}': {len(train_data)} training samples, {len(eval_data)} evaluation samples.")
    return train_data, eval_data

# --- STAGE 1: GENERATE DATA WITH GEMINI AND TRAIN MODEL V1 ---
print("\n\n--- STAGE 1: GENERATE DATA WITH GEMINI AND TRAIN MODEL V1 ---")

def generate_qa_with_gemini(doc_sample):
    q_prompt = f'From the text below, generate one specific, technical question.\n\nText: "{doc_sample}"\n\nQuestion:'
    question = generate_response_gemini(q_prompt)
    if not question: return None

    a_prompt = f'Based ONLY on the text provided, answer the following question.\n\nText: "{doc_sample}"\n\nQuestion: "{question}"\n\nAnswer:'
    answer = generate_response_gemini(a_prompt)
    if not answer: return None
    
    return {"instruction": question.strip(), "input": "", "output": answer.strip()}

# Generate dataset v1
dataset_v1_list = generate_dataset(generate_qa_with_gemini, 20, dataset_v1_path)
# Split and train model v1
if dataset_v1_list:
    train_ds_v1, eval_ds_v1 = prepare_and_save_splits(dataset_v1_list, "v1")
    model_v1_final_path = train_model(train_ds_v1, eval_ds_v1, base_model_id, model_v1_output_dir, max_steps=150)
else:
    print("Could not proceed with Stage 1 training as dataset is empty.")


# --- STAGE 2: GENERATE DATA WITH MODEL V1 AND TRAIN MODEL V2 ---
print("\n\n--- STAGE 2: GENERATE DATA WITH MODEL V1 AND TRAIN MODEL V2 ---")

# Load fine-tuned model v1
print("Loading fine-tuned model v1 for data generation...")
base_model_for_v1 = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
ft_model_v1 = PeftModel.from_pretrained(base_model_for_v1, model_v1_final_path)
tokenizer_v1 = AutoTokenizer.from_pretrained(base_model_id)
tokenizer_v1.pad_token = tokenizer_v1.eos_token

def generate_response_finetuned(model, tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    encodeds = tokenizer(text, return_tensors="pt").to('cuda')
    generated_ids = model.generate(**encodeds, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    new_tokens = generated_ids[0, encodeds['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def generate_qa_with_finetuned_model(doc_sample, model, tokenizer):
    # Use the fine-tuned model's instruction format to prompt it
    q_prompt = f"From the text below, generate one specific, technical question.\n\nText: \"{doc_sample}\""
    question = generate_response_finetuned(model, tokenizer, q_prompt)
    if not question: return None

    a_prompt = f"Based ONLY on the text provided, answer the following question.\n\nText: \"{doc_sample}\"\n\nQuestion: \"{question}\""
    answer = generate_response_finetuned(model, tokenizer, a_prompt)
    if not answer: return None
    
    return {"instruction": question.strip(), "input": "", "output": answer.strip()}


# Generate dataset v2 using model v1
dataset_v2_list = generate_dataset(generate_qa_with_finetuned_model, 120, dataset_v2_path, model=ft_model_v1, tokenizer=tokenizer_v1)

# Clean up model v1 from memory before training v2
del ft_model_v1
del base_model_for_v1
torch.cuda.empty_cache()

# Split and train model v2
if dataset_v2_list:
    train_ds_v2, eval_ds_v2 = prepare_and_save_splits(dataset_v2_list, "v2")
    model_v2_final_path = train_model(train_ds_v2, eval_ds_v2, base_model_id, model_v2_output_dir, max_steps=150)
else:
    print("Could not proceed with Stage 2 training as dataset is empty.")


print("\n\n--- STEP 8: COMPARATIVE INFERENCE ---")

# Load base model for inference
base_model_for_inference = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load both fine-tuned models
print("Loading Model v1 (trained on Gemini data)...")
ft_model_v1 = PeftModel.from_pretrained(base_model_for_inference, model_v1_final_path)

print("Loading Model v2 (trained on self-generated data)...")
ft_model_v2 = PeftModel.from_pretrained(base_model_for_inference, model_v2_final_path)


# --- Interactive Chat Loop ---
print("\n--- Starting Comparative Chat (type 'quit' to exit) ---")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
    
    print("\n--- Model v1 (Gemini Data) Response: ---")
    response_v1 = generate_response_finetuned(ft_model_v1, tokenizer, user_input)
    print(response_v1)
    
    print("\n--- Model v2 (Self-Generated Data) Response: ---")
    response_v2 = generate_response_finetuned(ft_model_v2, tokenizer, user_input)
    print(response_v2)
