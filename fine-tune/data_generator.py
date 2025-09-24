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

<<<<<<< HEAD
# --- SCHEMA HAS BEEN REMOVED ---
=======
# --- NEW: SCHEMA FOR GUIDED QUESTION GENERATION ---
SCHEMA_SUMMARY = """
- **Synthesis:** method (CVD, PVD, etc.), temperature, pressure, gases (carbon source, carrier).
- **Materials:** substrate (Si, quartz), catalyst (Fe, Ni, Co), precursors.
- **Outcomes/Morphology:** structure (forest, film), alignment, diameter, length, defects.
- **Characterization/Properties:** Raman (ID/IG ratio), electrical conductivity, mechanical strength.
"""
# Automatically parse the schema into a clean list of topics
SCHEMA_TOPICS = [
    line.split('**')[1].split(':')[0]
    for line in SCHEMA_SUMMARY.strip().splitlines()
    if line.strip().startswith('- **')
]
print(f"Data generator initialized with schema topics: {SCHEMA_TOPICS}")
# --- END NEW SECTION ---

>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9

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

<<<<<<< HEAD
# --- MODIFIED: Reverted to the original "expert persona" prompt ---
def generate_qa_gemini(doc_sample):
    """Generates a Q&A pair using Gemini with an expert persona."""
    q_prompt = f"""You are an expert CNT researcher reviewing a document. Based on the following text snippet, formulate one insightful and specific technical question that an advanced colleague might ask. The question should probe a key concept within the text.
=======
def generate_qa_gemini(doc_sample):
    """
    MODIFIED: Generates a Q&A pair using Gemini, guided by a randomly selected schema topic.
    """
    # Randomly select a topic from the schema for this Q&A pair
    topic = random.choice(SCHEMA_TOPICS)

    q_prompt = f"""You are an expert CNT researcher reviewing a document. Your task is to generate a specific technical question based on the provided text snippet.

**Focus your question specifically on the topic of: {topic}**

Use the text to formulate an insightful question that an advanced colleague might ask.
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9

**Text Snippet:**
"{doc_sample}"

**Generated Question:**"""
    question = _generate_response_gemini(q_prompt)
    if not question: return None

    a_prompt = f"""You are an expert CNT researcher answering a colleague's question. Use ONLY the provided context to write a comprehensive, well-structured answer. Synthesize the information rather than just extracting it. Do not add any preamble like 'Based on the text...'.
<<<<<<< HEAD

**Colleague's Question:**
"{question}"

**Context to Use:**
"{doc_sample}"

**Synthesized Answer:**"""
=======
**Colleague's Question:**\n"{question}"\n**Context to Use:**\n"{doc_sample}"\n**Synthesized Answer:**"""
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    answer = _generate_response_gemini(a_prompt)
    if not answer: return None
    return {"instruction": question.strip(), "input": "", "output": answer.strip()}

<<<<<<< HEAD
# --- MODIFIED: Reverted to the original "expert persona" prompt ---
def generate_qa_finetuned(doc_sample, model, tokenizer):
    """Generates a Q&A pair using a fine-tuned model with an expert persona."""
    q_prompt = f"You are an expert CNT researcher. From the text below, generate one specific, technical question.\n\nText: \"{doc_sample}\""
=======
def generate_qa_finetuned(doc_sample, model, tokenizer):
    """
    MODIFIED: Generates a Q&A pair using a fine-tuned model, guided by a randomly selected schema topic.
    """
    # Randomly select a topic from the schema for this Q&A pair
    topic = random.choice(SCHEMA_TOPICS)

    q_prompt = f"""You are an expert CNT researcher. From the text below, generate one specific, technical question focusing on the topic of **{topic}**.

Text: "{doc_sample}"
"""
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    question = generate_response(model, tokenizer, q_prompt)
    if not question: return None

    a_prompt = f"As an expert CNT researcher, use ONLY the provided text to answer the following question in a synthesized manner.\n\nQuestion: \"{question}\"\n\nContext: \"{doc_sample}\""
    answer = generate_response(model, tokenizer, a_prompt)
    if not answer: return None
    return {"instruction": question.strip(), "input": "", "output": answer.strip()}

<<<<<<< HEAD

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
=======
# --- MODIFIED: The definitive solution is implemented here ---
def generate_deep_qa(doc_sample, generator_model, tokenizer=None, generator_choice='gemini'):
    """Generates a high-level Q&A pair from a large chunk with surgical, context-aware truncation."""
    
    # Define the template for the instructions
    prompt_template = """CONTEXT:\n---\n{context}\n---\nBased on the entire CONTEXT above, generate one insightful, high-level question about the main findings or methodology, and provide a summary answer.\n\nYour output must be in this format:\nQUESTION: [Your insightful, high-level question here]\nANSWER: [Your summary answer here]"""

    if generator_choice != 'gemini':
        # --- NEW: Surgical Truncation Logic for HF Models ---
        # 1. Calculate the number of tokens used by the prompt template instructions
        template_without_context = prompt_template.format(context="")
        template_tokens = tokenizer(template_without_context, return_tensors="pt")['input_ids'][0]
        num_template_tokens = len(template_tokens)
        
        # 2. Determine the available space for the document context
        # We subtract a small buffer (e.g., 50 tokens) for safety
        max_context_tokens = tokenizer.model_max_length - num_template_tokens - 50
        
        # 3. Tokenize and truncate the doc_sample itself
        doc_tokens = tokenizer(doc_sample, return_tensors="pt")['input_ids'][0]
        truncated_doc_tokens = doc_tokens[:max_context_tokens]
        
        # 4. Decode the truncated tokens back into a string
        truncated_doc_sample = tokenizer.decode(truncated_doc_tokens, skip_special_tokens=True)
        
        # 5. Build the final, safely-sized prompt
        final_prompt = prompt_template.format(context=truncated_doc_sample)
        full_generation = generate_response(generator_model, tokenizer, final_prompt)
    else:
        # For Gemini, which has a very large context window, we can use the original method
        final_prompt = prompt_template.format(context=doc_sample)
        full_generation = _generate_response_gemini(final_prompt)
    
    # --- The rest of the function is unchanged ---
    if not full_generation or "ANSWER:" not in full_generation:
        return None
        
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    try:
        parts = re.split(r'\bANSWER:\b', full_generation, maxsplit=1)
        question = parts[0].replace("QUESTION:", "").strip()
        answer = parts[1].strip()
<<<<<<< HEAD
        if len(question) < 15 or len(answer) < 15: return None
        return {"instruction": question, "input": "", "output": answer.strip()}
    except (IndexError, ValueError): return None

# --- MASTER ORCHESTRATOR FUNCTION (with enhanced progress reporting) ---
def create_combined_dataset_for_stage(config, raw_text, stage_tag, gen_funcs, generator_choice, model=None, tokenizer=None):
    """Orchestrates the creation of chunk, deep, and combined datasets for a given stage."""
=======
        
        if len(question) < 15 or len(answer) < 15:
            return None
            
        return {"instruction": question, "input": "", "output": answer.strip()}
    except (IndexError, ValueError):
        return None

# --- MASTER ORCHESTRATOR FUNCTION (UNCHANGED) ---

def create_combined_dataset_for_stage(config, raw_text, stage_tag, gen_funcs, generator_choice, model=None, tokenizer=None):
    """
    Orchestrates the creation of chunk, deep, and combined datasets for a given stage.
    """
    # Determine file paths based on stage_tag ('v1' or 'v2')
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    paths = {
        'chunks': config.DATASET_V1_CHUNKS_PATH if stage_tag == 'v1' else config.DATASET_V2_CHUNKS_PATH,
        'deep': config.DATASET_V1_DEEP_PATH if stage_tag == 'v1' else config.DATASET_V2_DEEP_PATH,
        'combined': config.DATASET_V1_COMBINED_PATH if stage_tag == 'v1' else config.DATASET_V2_COMBINED_PATH
    }
<<<<<<< HEAD
    if os.path.exists(paths['combined']):
        print(f"✅ Found existing dataset '{os.path.basename(paths['combined'])}'.")
        print("--- SKIPPING DATA GENERATION ---")
        with open(paths['combined'], "r") as f:
            return json.load(f)
=======

    # if os.path.exists(paths['combined']):
    #     print(f"Final combined dataset '{paths['combined']}' already exists. Loading.")
    #     with open(paths['combined'], "r") as f:
    #         return json.load(f)

    if os.path.exists(paths['combined']):
        print(f"✅ Final combined dataset '{os.path.basename(paths['combined'])}' already exists. Loading from file and skipping generation.")
        with open(paths['combined'], "r") as f:
            return json.load(f) # <-- It immediately returns the loaded data and exits the function.
    # --- END OF CRITICAL LOGIC ---
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9

    if not raw_text:
        print("Error: Raw text is empty. Cannot generate datasets.")
        return []

<<<<<<< HEAD
    print(f"\n--- No existing dataset found. Generating new dataset for Stage '{stage_tag}' ---")
    
    # --- NEW: Logic for unified progress reporting ---
    total_samples_to_generate = config.NUM_SAMPLES_TO_GENERATE_CHUNKS + config.NUM_SAMPLES_TO_GENERATE_DEEP
    current_sample_count = 0
    # --- END NEW LOGIC ---

    # --- 1. Generate Chunk-based Dataset ---
    print("\n--- Generating chunk-based dataset ---")
    num_samples_chunks = config.NUM_SAMPLES_TO_GENERATE_CHUNKS
=======
    # --- 1. Generate Chunk-based Dataset ---
    print("\n--- Generating chunk-based dataset ---")
    num_samples_chunks = config.NUM_SAMPLES_TO_GENERATE_CHUNKS
    
    # Use a lambda to pass a new random doc_sample on each call
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    chunk_generator_lambda = lambda: gen_funcs['chunks'](
        doc_sample=" ".join(raw_text[random.randint(0, len(raw_text) - size):][:size].split()),
        **({'model': model, 'tokenizer': tokenizer} if model else {})
    )
<<<<<<< HEAD
=======
    
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    data_chunks = []
    for size in config.CHUNK_SIZES:
        iterations_per_size = num_samples_chunks // len(config.CHUNK_SIZES)
        print(f"Generating {iterations_per_size} samples for chunk size {size}...")
        for i in range(iterations_per_size):
<<<<<<< HEAD
            # MODIFIED: Update and print progress
            current_sample_count += 1
            print(f"  Generating sample {current_sample_count}/{total_samples_to_generate}...")
            qa_pair = chunk_generator_lambda()
            if qa_pair:
                data_chunks.append(qa_pair)
=======
            qa_pair = chunk_generator_lambda()
            if qa_pair:
                data_chunks.append(qa_pair)

>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    with open(paths['chunks'], "w") as f: json.dump(data_chunks, f, indent=4)
    print(f"Saved {len(data_chunks)} chunk samples to '{paths['chunks']}'")

    # --- 2. Generate Deep (Summary) Dataset ---
    print("\n--- Generating deep summary dataset ---")
    deep_chunk_size = config.DEEP_CHUNK_SIZE
<<<<<<< HEAD
=======

>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    deep_generator_lambda = lambda: gen_funcs['deep'](
        doc_sample=" ".join(raw_text[random.randint(0, len(raw_text) - deep_chunk_size):][:deep_chunk_size].split()),
        generator_model=model if model else gemini_model,
        tokenizer=tokenizer,
        generator_choice=generator_choice
    )
<<<<<<< HEAD
    
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
=======

    data_deep = _generate_and_save_partial_dataset(
        paths['deep'],
        num_samples=config.NUM_SAMPLES_TO_GENERATE_DEEP,
        generation_function=deep_generator_lambda
    )
>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    
    # --- 3. Combine and Save ---
    print("\n--- Combining datasets ---")
    combined_data = data_chunks + data_deep
    random.shuffle(combined_data)
<<<<<<< HEAD
    with open(paths['combined'], "w") as f:
        json.dump(combined_data, f, indent=4)
=======

    with open(paths['combined'], "w") as f:
        json.dump(combined_data, f, indent=4)

>>>>>>> a74b644835d46650b0ab91769ffcb607430a46c9
    print(f"Successfully created combined dataset with {len(combined_data)} samples at '{paths['combined']}'.")
    return combined_data