import os
import sys
import argparse
import importlib.util
import torch

from data_processor import process_pdfs_with_unstructured
from data_generator import create_combined_dataset_for_stage, generate_qa_gemini, generate_qa_finetuned, generate_deep_qa
from model_trainer import run_training
from model_utils import load_base_model_and_tokenizer, load_peft_model
from evaluator import run_traditional_evaluation, run_llm_judge_evaluation, evaluate_datasets, _plot_loss_curves

def setup_directories(config):
    """Creates all necessary output directories defined in the config."""
    print("--- Setting up project directories ---")
    dir_paths = [
        config.SCRATCH_DIR,
        config.OUTPUT_DIR_VISUALIZATIONS,
        config.OUTPUT_DIR_EVALUATION,
        config.GENERATED_DATA_DIR,
        config.CACHE_DIR_UNSTRUCTURED
    ]
    if hasattr(config, 'LOCAL_MODEL_SAVE_DIR'):
        dir_paths.append(config.LOCAL_MODEL_SAVE_DIR)
        
    for dir_path in dir_paths:
        try:
            if not os.path.exists(dir_path):
                # Use exist_ok=True to prevent race conditions in parallel environments
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
        except PermissionError:
            print("\n" + "="*80)
            print(f"FATAL ERROR: Permission denied when trying to create directory: {dir_path}")
            print("This is a common issue on HPC clusters.")
            print("Please check that the `SCRATCH_DIR` in your config file points to a valid,")
            print("writable location for your user account on this cluster.")
            print("Run `echo $SCRATCH` or `echo $WORK` in your terminal to find the correct path.")
            print("="*80 + "\n")
            # Exit the script because it cannot continue without storage
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while creating directory {dir_path}: {e}")
            sys.exit(1)

def load_config(config_name):
    """Dynamically loads a configuration file."""
    config_path = f"{config_name}.py"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    print(f"Successfully loaded configuration from '{config_path}'")
    return config_module

def main():
    parser = argparse.ArgumentParser(description="Run a two-stage LLM fine-tuning and evaluation pipeline.")
    parser.add_argument('--config', type=str, required=True, help='Name of the configuration file (without .py extension)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # --- NEW: Derive a short, unique name for this experiment run ---
    experiment_name = args.config.replace("config_", "") # e.g., "GQQ" or "QQQ"
    print(f"\n--- Running Experiment: {experiment_name.upper()} ---")
    
    setup_directories(config)

    print("\n--- STEP 1: LOAD AND PROCESS ALL PDFS WITH UNSTRUCTURED ---")
    processed_docs = process_pdfs_with_unstructured(config.PDF_FOLDER_PATH, config.CACHE_DIR_UNSTRUCTURED)
    raw_text = "\n\n--- END OF DOCUMENT ---\n\n".join(processed_docs.values())

    if not raw_text:
        print("No text extracted from PDFs. Exiting.")
        return
    print(f"Total characters extracted from all PDFs: {len(raw_text)}")

    print("\n\n--- STAGE 1: GENERATE DATA WITH BASE MODEL AND TRAIN V1 ---")
    if config.STAGE1_GENERATOR_CHOICE == 'gemini':
        stage1_gen_funcs = {'chunks': generate_qa_gemini, 'deep': generate_deep_qa}
        dataset_v1_list = create_combined_dataset_for_stage(config=config, raw_text=raw_text, stage_tag='v1', gen_funcs=stage1_gen_funcs, generator_choice='gemini')
    else:
        print(f"Using base model '{config.STAGE1_GENERATOR_CHOICE}' for Stage 1 data generation.")
        model_choice = config.STAGE1_GENERATOR_CHOICE
        base_model_id = config.AVAILABLE_MODELS[model_choice]
        base_model_gen, tokenizer_gen = load_base_model_and_tokenizer(base_model_id, config.BNB_CONFIG, model_choice, config.CHAT_TEMPLATES)
        stage1_gen_funcs = {'chunks': generate_qa_finetuned, 'deep': generate_deep_qa}
        dataset_v1_list = create_combined_dataset_for_stage(config=config, raw_text=raw_text, stage_tag='v1', gen_funcs=stage1_gen_funcs, generator_choice=model_choice, model=base_model_gen, tokenizer=tokenizer_gen)
        print("Stage 1 generation complete. Releasing model from memory before training.")
        del base_model_gen, tokenizer_gen
        torch.cuda.empty_cache()
    
    model_v1_final_path, history_v1 = (None, None)
    if dataset_v1_list:
        model_v1_final_path, history_v1 = run_training(config, dataset_v1_list, model_choice=config.STAGE1_TRAINING_CHOICE, output_dir=config.MODEL_V1_OUTPUT_DIR)
        if history_v1: 
            # MODIFIED: Pass the experiment_name to the plotting function
            _plot_loss_curves(history_v1, "v1", config.OUTPUT_DIR_VISUALIZATIONS, experiment_name)
    else:
        print("Could not proceed with Stage 1 training as dataset is empty.")

    print("\n\n--- STAGE 2: GENERATE DATA WITH MODEL V1 AND TRAIN V2 ---")
    model_v2_final_path, history_v2 = (None, None)
    if model_v1_final_path and os.path.exists(os.path.join(model_v1_final_path)):
        print("Loading fine-tuned model v1 for data generation...")
        model_choice = config.STAGE1_TRAINING_CHOICE
        base_model_id = config.AVAILABLE_MODELS[model_choice]
        base_model, tokenizer_v1 = load_base_model_and_tokenizer(base_model_id, config.BNB_CONFIG, model_choice, config.CHAT_TEMPLATES)
        ft_model_v1 = load_peft_model(base_model, model_v1_final_path)
        stage2_gen_funcs = {'chunks': generate_qa_finetuned, 'deep': generate_deep_qa}
        dataset_v2_list = create_combined_dataset_for_stage(config=config, raw_text=raw_text, stage_tag='v2', gen_funcs=stage2_gen_funcs, generator_choice=config.STAGE2_TRAINING_CHOICE, model=ft_model_v1, tokenizer=tokenizer_v1)
        del ft_model_v1, base_model, tokenizer_v1
        torch.cuda.empty_cache()
        if dataset_v2_list:
            model_v2_final_path, history_v2 = run_training(config, dataset_v2_list, model_choice=config.STAGE2_TRAINING_CHOICE, output_dir=config.MODEL_V2_OUTPUT_DIR)
            if history_v2: 
                # MODIFIED: Pass the experiment_name to the plotting function
                _plot_loss_curves(history_v2, "v2", config.OUTPUT_DIR_VISUALIZATIONS, experiment_name)
        else:
            print("Could not proceed with Stage 2 training as dataset is empty.")
    else:
        print("Skipping Stage 2 because Stage 1 model was not trained or path not found.")

    if model_v1_final_path and model_v2_final_path:
        print("\n\n--- FINAL EVALUATION STAGE ---")
        model_choice = config.STAGE2_TRAINING_CHOICE
        base_model_id = config.AVAILABLE_MODELS[model_choice]
        base_model_for_eval, tokenizer = load_base_model_and_tokenizer(base_model_id, config.BNB_CONFIG, model_choice, config.CHAT_TEMPLATES)
        print("Loading Model v1 for evaluation...")
        ft_model_v1_eval = load_peft_model(base_model_for_eval, model_v1_final_path)
        print("Loading Model v2 for evaluation...")
        ft_model_v2_eval = load_peft_model(base_model_for_eval, model_v2_final_path)
        
        # MODIFIED: Pass the experiment_name to the evaluation function
        run_traditional_evaluation(config, ft_model_v1_eval, ft_model_v2_eval, tokenizer, experiment_name)
        run_llm_judge_evaluation(config, ft_model_v1_eval, ft_model_v2_eval, tokenizer)
        evaluate_datasets(config)
    else:
        print("\nSkipping final evaluation as one or both models were not trained.")
        
    print(f"\n--- EXPERIMENT {experiment_name.upper()} COMPLETE ---")

if __name__ == "__main__":
    main()