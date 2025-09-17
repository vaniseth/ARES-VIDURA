# config_GMM.py
import torch
import transformers
import os

# ======================================================================================
# --- CORE EXPERIMENT SETTINGS ---
# ======================================================================================
# Choices for STAGE1_GENERATOR_CHOICE: "gemini"
# Choices for JUDGE_MODEL_CHOICE: "gemini"
# Choices for STAGE1_TRAINING_CHOICE / STAGE2_TRAINING_CHOICE: "mistral"
AVAILABLE_MODELS = {
    "mistral": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

STAGE1_GENERATOR_CHOICE = "mistral"
STAGE1_TRAINING_CHOICE = "mistral"
STAGE2_TRAINING_CHOICE = "mistral"
JUDGE_MODEL_CHOICE = "gemini"

# ======================================================================================
# --- DYNAMIC FILENAME GENERATION (DO NOT EDIT) ---
# ======================================================================================
s1_gen_tag = f"gen_{STAGE1_GENERATOR_CHOICE}"
s1_train_tag = f"train_{STAGE1_TRAINING_CHOICE}"
S1_TAG = f"{s1_gen_tag}_{s1_train_tag}_v1"
s2_train_tag = f"train_{STAGE2_TRAINING_CHOICE}"
S2_TAG = f"{S1_TAG}_self-improve_{s2_train_tag}_v2"

# ======================================================================================
# --- FILE PATHS AND DIRECTORIES (DO NOT EDIT) ---
# ======================================================================================
PDF_FOLDER_PATH = "../DataSets/"
SCRATCH_DIR = f"/tmp/{os.getenv('USER', 'user')}/cnt_finetuning"
CACHE_DIR_UNSTRUCTURED = "unstructured_output" # <-- ADD THIS LINE
OUTPUT_DIR_VISUALIZATIONS = "visualizations"
OUTPUT_DIR_EVALUATION = "evaluation_outputs"
GENERATED_DATA_DIR = "generated_training_data"
EVALUATION_DATA_PATH = "model-comp-adv.json"

# --- NEW: Updated Dynamic Dataset Paths ---
DATASET_V1_CHUNKS_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S1_TAG}_chunks.json")
DATASET_V1_DEEP_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S1_TAG}_deep.json")
DATASET_V1_COMBINED_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S1_TAG}_combined.json")

DATASET_V2_CHUNKS_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S2_TAG}_chunks.json")
DATASET_V2_DEEP_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S2_TAG}_deep.json")
DATASET_V2_COMBINED_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S2_TAG}_combined.json")
# --- END OF NEW SECTION ---

# Dynamic Model & Eval Paths
MODEL_V1_OUTPUT_DIR = os.path.join(SCRATCH_DIR, f"sft_model_{S1_TAG}")
MODEL_V2_OUTPUT_DIR = os.path.join(SCRATCH_DIR, f"sft_model_{S2_TAG}")
LLM_JUDGE_RESULTS_PATH = os.path.join(OUTPUT_DIR_EVALUATION, f"llm_judge_results_{S2_TAG}.csv")
DATASET_JUDGE_RESULTS_PATH = os.path.join(OUTPUT_DIR_EVALUATION, f"dataset_judge_results_{S2_TAG}.csv")

# ======================================================================================
# --- HYPERPARAMETERS & CONFIGS ---
# ======================================================================================
# Data Generation
NUM_SAMPLES_TO_GENERATE_CHUNKS = 10
NUM_SAMPLES_TO_GENERATE_DEEP = 5
CHUNK_SIZES = [1024, 2048]
DEEP_CHUNK_SIZE = 4096

# Training
MAX_TRAINING_STEPS = 300
LOGGING_STEPS = 10
EVAL_STEPS = 20
SAVE_STEPS = 20
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LEARNING_RATE = 2.5e-5

# Hardware & Quantization
BNB_CONFIG = transformers.BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Chat Templates (no changes here)
MISTRAL_CHAT_TEMPLATE = (
    "<s>[INST] {user_content} [/INST] {assistant_content} </s>"
)
CHAT_TEMPLATES = {"mistral": MISTRAL_CHAT_TEMPLATE}