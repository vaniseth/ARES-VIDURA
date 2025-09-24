<<<<<<< HEAD
# config_GMM.py
=======
# config_QMQ.py
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
import torch
import transformers
import os

# ======================================================================================
# --- CORE EXPERIMENT SETTINGS ---
# ======================================================================================
<<<<<<< HEAD
# Choices for STAGE1_GENERATOR_CHOICE: "gemini"
# Choices for JUDGE_MODEL_CHOICE: "gemini"
# Choices for STAGE1_TRAINING_CHOICE / STAGE2_TRAINING_CHOICE: "mistral"
AVAILABLE_MODELS = {
=======
AVAILABLE_MODELS = {
    "mistral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    "quokka": "Xianjun/Quokka-13b-instruct"
}

STAGE1_GENERATOR_CHOICE = "gemini"
STAGE1_TRAINING_CHOICE = "quokka"
STAGE2_TRAINING_CHOICE = "quokka"
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
<<<<<<< HEAD
PDF_FOLDER_PATH = "../DataSets/"
=======
PDF_FOLDER_PATH = "../DataSets/" # Use relative path for portability
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9

# --- THIS IS THE CORRECTED TWO-DIRECTORY SETUP ---
# Use a large, temporary scratch space on the cluster for training checkpoints
# Replace '/scratch/' with your cluster's actual scratch path if different
SCRATCH_DIR = f"/tmp/{os.getenv('USER', 'user')}/cnt_finetuning"

# Define the final, local directory for clean model adapters
LOCAL_MODEL_SAVE_DIR = "trained_models" 
# --- END OF CORRECTION ---

<<<<<<< HEAD
CACHE_DIR_UNSTRUCTURED = "unstructured_output" # <-- ADD THIS LINE
=======
CACHE_DIR_UNSTRUCTURED = "unstructured_output"
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
OUTPUT_DIR_VISUALIZATIONS = "visualizations"
OUTPUT_DIR_EVALUATION = "evaluation_outputs"
GENERATED_DATA_DIR = "generated_training_data"
EVALUATION_DATA_PATH = "model-comp-adv.json"

<<<<<<< HEAD
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
=======
# These paths will now correctly use the temporary SCRATCH_DIR for training output
MODEL_V1_OUTPUT_DIR = os.path.join(SCRATCH_DIR, f"sft_model_{S1_TAG}")
MODEL_V2_OUTPUT_DIR = os.path.join(SCRATCH_DIR, f"sft_model_{S2_TAG}")

# --- Dataset and Eval paths are unchanged ---
DATASET_V1_CHUNKS_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S1_TAG}_chunks.json")
DATASET_V1_DEEP_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S1_TAG}_deep.json")
DATASET_V1_COMBINED_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S1_TAG}_combined.json")
DATASET_V2_CHUNKS_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S2_TAG}_chunks.json")
DATASET_V2_DEEP_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S2_TAG}_deep.json")
DATASET_V2_COMBINED_PATH = os.path.join(GENERATED_DATA_DIR, f"dataset_{S2_TAG}_combined.json")
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
LLM_JUDGE_RESULTS_PATH = os.path.join(OUTPUT_DIR_EVALUATION, f"llm_judge_results_{S2_TAG}.csv")
DATASET_JUDGE_RESULTS_PATH = os.path.join(OUTPUT_DIR_EVALUATION, f"dataset_judge_results_{S2_TAG}.csv")

# ======================================================================================
# --- HYPERPARAMETERS & CONFIGS ---
# ======================================================================================
<<<<<<< HEAD
# Data Generation
NUM_SAMPLES_TO_GENERATE_CHUNKS = 360
NUM_SAMPLES_TO_GENERATE_DEEP = 140
CHUNK_SIZES = [1024, 768, 512]
DEEP_CHUNK_SIZE = 1024

# Training
MAX_TRAINING_STEPS = 300
LOGGING_STEPS = 10
=======
# Data Generation (Corrected for Quokka's context limit)
NUM_SAMPLES_TO_GENERATE_CHUNKS = 10
NUM_SAMPLES_TO_GENERATE_DEEP = 5
CHUNK_SIZES = [512, 800]      # CORRECTED: Smaller chunks for Quokka
DEEP_CHUNK_SIZE = 900         # CORRECTED: Smaller deep chunk for Quokka

# Training (Corrected for stability and smoother loss)
MAX_TRAINING_STEPS = 150
LOGGING_STEPS = 10            # CORRECTED: For smoother loss plot
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
EVAL_STEPS = 20
SAVE_STEPS = 20
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
<<<<<<< HEAD
LEARNING_RATE = 2.5e-5
=======
LEARNING_RATE = 1.5e-5        # CORRECTED: For more stable training
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9

# Hardware & Quantization
BNB_CONFIG = transformers.BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Chat Templates
LLAMA2_CHAT_TEMPLATE = ("{% for message in messages %}{% if message['role'] == 'user' %}<s>[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %} {{ message['content'] }}</s>{% endif %}{% endfor %}")
MISTRAL_CHAT_TEMPLATE = ("{% for message in messages %}{% if message['role'] == 'user' %}<s>[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}{% endfor %}")
<<<<<<< HEAD
CHAT_TEMPLATES = {
    "quokka": LLAMA2_CHAT_TEMPLATE,
    "mistral": MISTRAL_CHAT_TEMPLATE,
}
=======
CHAT_TEMPLATES = {"quokka": LLAMA2_CHAT_TEMPLATE, "mistral": MISTRAL_CHAT_TEMPLATE}
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
