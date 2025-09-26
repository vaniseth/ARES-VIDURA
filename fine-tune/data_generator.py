import os
import json
import random
import time
import re
import google.generativeai as genai
from dotenv import load_dotenv
from model_utils import generate_response

# --- GEMINI API SETUP ---
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-2.5-pro')
except Exception as e:
    print(f"Could not configure Gemini. Ensure GOOGLE_API_KEY is set. Error: {e}")
    gemini_model = None

# --- SCHEMA HAS BEEN REMOVED ---

# --- HELPER FUNCTIONS ---
def _generate_response_gemini(prompt):
    """Internal function to call Gemini API with error handling."""
    if not gemini_model:
        print("Gemini model not initialized. Skipping API call.")
        return None
    try:
        response = gemini_model.generate_content(prompt)
        return response.text if response.parts else None
    except Exception as e:
        print(f"Gemini API error: {e}")
        time.sleep(5)
        return None

def _generate_and_save_partial_dataset(output_filename, num_samples, generation_function, **kwargs):
    """Helper to generate and save one type of dataset (e.g., chunks or deep)."""
    if os.path.exists(output_filename):
        print(f"Partial dataset '{output_filename}' already exists. Loading.")
        with open(output_filename, "r") as f:
            return json.load(f)

    data = []
    for i in range(num_samples):
        print(f"  Generating sample {i + 1}/{num_samples} for {os.path.basename(output_filename)}...")
        qa_pair = generation_function(**kwargs)
        if qa_pair:
            data.append(qa_pair)

    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  Successfully generated {len(data)} samples for '{output_filename}'.")
    return data

# --- SPECIALIZED QA GENERATION FUNCTIONS ---

# --- MODIFIED: Reverted to the original "expert persona" prompt ---
def generate_qa_gemini(doc_sample):
    """Generates a Q&A pair using Gemini with an expert persona."""
    q_prompt = f"""You are an expert CNT researcher reviewing a document. Based on the following text snippet, formulate one insightful and specific technical question that an advanced colleague might ask. The question should probe a key concept within the text.

**Text Snippet:**
"{doc_sample}"

**Generated Question:**"""
    question = _generate_response_gemini(q_prompt)
    if not question: return None

    a_prompt = f"""You are an expert CNT researcher answering a colleague's question. Use ONLY the provided context to write a comprehensive, well-structured answer. Synthesize the information rather than just extracting it. Do not add any preamble like 'Based on the text...'.

**Colleague's Question:**
"{question}"

**Context to Use:**
"{doc_sample}"

**Synthesized Answer:**"""
    answer = _generate_response_gemini(a_prompt)
    if not answer: return None
    return {"instruction": question.strip(), "input": "", "output": answer.strip()}

# --- MODIFIED: Reverted to the original "expert persona" prompt ---
def generate_qa_finetuned(doc_sample, model, tokenizer):
    """Generates a Q&A pair using a fine-tuned model with an expert persona."""
    q_prompt = f"You are an expert CNT researcher. From the text below, generate one specific, technical question.\n\nText: \"{doc_sample}\""
    question = generate_response(model, tokenizer, q_prompt)
    if not question: return None

    a_prompt = f"As an expert CNT researcher, use ONLY the provided text to answer the following question in a synthesized manner.\n\nQuestion: \"{question}\"\n\nContext: \"{doc_sample}\""
    answer = generate_response(model, tokenizer, a_prompt)
    if not answer: return None
    return {"instruction": question.strip(), "input": "", "output": answer.strip()}


# --- UNCHANGED: The deep_qa function remains as a separate, valuable component ---
def generate_deep_qa(doc_sample, generator_model, tokenizer=None, generator_choice='gemini'):
    """Generates a high-level Q&A pair from a large chunk with surgical, context-aware truncation."""
    prompt_template = """CONTEXT:\n---\n{context}\n---\nBased on the entire CONTEXT above, generate one insightful, high-level question about the main findings or methodology, and provide a summary answer.\n\nYour output must be in this format:\nQUESTION: [Your insightful, high-level question here]\nANSWER: [Your summary answer here]"""
    if generator_choice != 'gemini':
        template_without_context = prompt_template.format(context="")
        template_tokens = tokenizer(template_without_context, return_tensors="pt")['input_ids'][0]
        num_template_tokens = len(template_tokens)
        max_context_tokens = tokenizer.model_max_length - num_template_tokens - 50
        doc_tokens = tokenizer(doc_sample, return_tensors="pt")['input_ids'][0]
        truncated_doc_tokens = doc_tokens[:max_context_tokens]
        truncated_doc_sample = tokenizer.decode(truncated_doc_tokens, skip_special_tokens=True)
        final_prompt = prompt_template.format(context=truncated_doc_sample)
        full_generation = generate_response(generator_model, tokenizer, final_prompt)
    else:
        final_prompt = prompt_template.format(context=doc_sample)
        full_generation = _generate_response_gemini(final_prompt)
    if not full_generation or "ANSWER:" not in full_generation: return None
    try:
        parts = re.split(r'\bANSWER:\b', full_generation, maxsplit=1)
        question = parts[0].replace("QUESTION:", "").strip()
        answer = parts[1].strip()
        if len(question) < 15 or len(answer) < 15: return None
        return {"instruction": question, "input": "", "output": answer.strip()}
    except (IndexError, ValueError): return None

# --- MASTER ORCHESTRATOR FUNCTION (with enhanced progress reporting) ---
def create_combined_dataset_for_stage(config, raw_text, stage_tag, gen_funcs, generator_choice, model=None, tokenizer=None):
    """Orchestrates the creation of chunk, deep, and combined datasets for a given stage."""
    paths = {
        'chunks': config.DATASET_V1_CHUNKS_PATH if stage_tag == 'v1' else config.DATASET_V2_CHUNKS_PATH,
        'deep': config.DATASET_V1_DEEP_PATH if stage_tag == 'v1' else config.DATASET_V2_DEEP_PATH,
        'combined': config.DATASET_V1_COMBINED_PATH if stage_tag == 'v1' else config.DATASET_V2_COMBINED_PATH
    }
    if os.path.exists(paths['combined']):
        print(f"✅ Found existing dataset '{os.path.basename(paths['combined'])}'.")
        print("--- SKIPPING DATA GENERATION ---")
        with open(paths['combined'], "r") as f:
            return json.load(f)

    if not raw_text:
        print("Error: Raw text is empty. Cannot generate datasets.")
        return []

    print(f"\n--- No existing dataset found. Generating new dataset for Stage '{stage_tag}' ---")
    
    # --- NEW: Logic for unified progress reporting ---
    total_samples_to_generate = config.NUM_SAMPLES_TO_GENERATE_CHUNKS + config.NUM_SAMPLES_TO_GENERATE_DEEP
    current_sample_count = 0
    # --- END NEW LOGIC ---

    # --- 1. Generate Chunk-based Dataset ---
    print("\n--- Generating chunk-based dataset ---")
    num_samples_chunks = config.NUM_SAMPLES_TO_GENERATE_CHUNKS
    chunk_generator_lambda = lambda: gen_funcs['chunks'](
        doc_sample=" ".join(raw_text[random.randint(0, len(raw_text) - size):][:size].split()),
        **({'model': model, 'tokenizer': tokenizer} if model else {})
    )
    data_chunks = []
    for size in config.CHUNK_SIZES:
        iterations_per_size = num_samples_chunks // len(config.CHUNK_SIZES)
        print(f"Generating {iterations_per_size} samples for chunk size {size}...")
        for i in range(iterations_per_size):
            # MODIFIED: Update and print progress
            current_sample_count += 1
            print(f"  Generating sample {current_sample_count}/{total_samples_to_generate}...")
            qa_pair = chunk_generator_lambda()
            if qa_pair:
                data_chunks.append(qa_pair)
    with open(paths['chunks'], "w") as f: json.dump(data_chunks, f, indent=4)
    print(f"Saved {len(data_chunks)} chunk samples to '{paths['chunks']}'")

    # --- 2. Generate Deep (Summary) Dataset ---
    print("\n--- Generating deep summary dataset ---")
    deep_chunk_size = config.DEEP_CHUNK_SIZE
    deep_generator_lambda = lambda: gen_funcs['deep'](
        doc_sample=" ".join(raw_text[random.randint(0, len(raw_text) - deep_chunk_size):][:deep_chunk_size].split()),
        generator_model=model if model else gemini_model,
        tokenizer=tokenizer,
        generator_choice=generator_choice
    )
    
    # MODIFIED: Re-implement the loop here to allow for unified progress reporting
    data_deep = []
    for i in range(config.NUM_SAMPLES_TO_GENERATE_DEEP):
        current_sample_count += 1
        print(f"  Generating sample {current_sample_count}/{total_samples_to_generate} (type: deep)...")
        qa_pair = deep_generator_lambda()
        if qa_pair:
            data_deep.append(qa_pair)
    with open(paths['deep'], "w") as f: json.dump(data_deep, f, indent=4)
    print(f"Saved {len(data_deep)} deep samples to '{paths['deep']}'")
    
    # --- 3. Combine and Save ---
    print("\n--- Combining datasets ---")
    combined_data = data_chunks + data_deep
    random.shuffle(combined_data)
    with open(paths['combined'], "w") as f:
        json.dump(combined_data, f, indent=4)
    print(f"Successfully created combined dataset with {len(combined_data)} samples at '{paths['combined']}'.")
    return combined_data