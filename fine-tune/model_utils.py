import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# --- MODIFIED: Function now accepts model_choice and chat_templates to apply the correct template ---
def load_base_model_and_tokenizer(model_id, bnb_config, model_choice, chat_templates):
    """Loads a base model and tokenizer, applying a custom chat template if specified."""
    hf_token = os.getenv("HF_API_KEY_HOME")
    if not hf_token:
        print("Warning: HF_API_KEY_HOME not found in .env file. Downloads may fail for gated models.")

    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    # --- NEW: Logic to set the chat template dynamically ---
    if model_choice in chat_templates:
        tokenizer.chat_template = chat_templates[model_choice]
        print(f"Applied custom chat template for '{model_choice}'.")
    else:
        print(f"Warning: No custom chat template found for '{model_choice}'. Using model's default.")
    # --- END NEW SECTION ---

    return model, tokenizer

def load_peft_model(base_model, peft_model_path):
    """Loads a PEFT model from a given path and attaches it to the base model."""
    print(f"Loading PEFT adapter from: {peft_model_path}")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    return model

def generate_response(model, tokenizer, prompt, max_new_tokens=1024):
    """Generates a response from a loaded model."""
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    # --- NEW: Safety logic for max_length ---
    # Define a practical upper limit for generation context length
    PRACTICAL_MAX_LENGTH = 4096 
    
    # Check if the tokenizer's max length is a reasonable number.
    # If it's a huge placeholder number, use our practical limit instead.
    model_max_len = tokenizer.model_max_length
    if model_max_len > 100000: # A check for placeholder values like 1e30
        print(f"Warning: Tokenizer's model_max_length ({model_max_len}) is excessively large. Capping at {PRACTICAL_MAX_LENGTH}.")
        effective_max_length = PRACTICAL_MAX_LENGTH
    else:
        effective_max_length = model_max_len
    # --- END NEW LOGIC ---

    inputs = tokenizer(
        text, 
        return_tensors="pt",
        truncation=True,
        max_length=effective_max_length # MODIFIED: Use the new safe value
    ).to('cuda')
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    new_tokens = generated_ids[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)