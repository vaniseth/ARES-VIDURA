import os
import sys
import json
import argparse
import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate
from sentence_transformers import SentenceTransformer, util
import importlib.util

# --- Import from your existing project files ---
from model_utils import load_base_model_and_tokenizer, load_peft_model, generate_response
from data_generator import _generate_response_gemini

def load_all_configs(config_names):
    """Loads multiple configuration files into a dictionary."""
    configs = {}
    for name in config_names:
        config_path = f"{name}.py"
        if not os.path.exists(config_path):
            print(f"Warning: Configuration file '{config_path}' not found. Skipping.")
            continue
        
        spec = importlib.util.spec_from_file_location(name, config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        configs[name] = config_module
        print(f"Successfully loaded configuration from '{config_path}'")
    return configs

def _get_llm_judge_scoring_prompt(question, response):
    """Creates a prompt for a judge LLM to score a single response."""
    return f"""You are a meticulous and impartial evaluator for a scientific research context, specifically Material Science. Your task is to score a single response to a given question.
**Evaluation Criteria (Score 1-10):**
1.  **Accuracy & Factual Grounding:** Is the answer correct, non-speculative, and grounded in scientific principles? (1=Pure speculation, 10=Perfectly accurate)
2.  **Helpfulness & Completeness:** How well does the response directly and comprehensively answer the question? (1=Not helpful, 10=Extremely helpful)
3.  **Clarity & Persona:** Is the response clear, well-structured, and written in the persona of an expert? (1=Generic chatbot, 10=Authentic expert)

**Question:**
{question}

**Response to Score:**
{response}

**Your Task:**
Provide a score (1-10) for each criterion. Then, provide a brief justification for your scores. Finally, give an overall score. Output ONLY a JSON object with the following structure:
{{
  "scores": {{
    "accuracy": <score>,
    "helpfulness": <score>,
    "clarity": <score>
  }},
  "justification": "<Your brief analysis here>",
  "overall_score": <A single score from 1-10 representing your overall impression>
}}"""

def generate_summary_plots(summary_df, output_dir):
    """Generates and saves summary plots from the final summary DataFrame."""
    if summary_df.empty:
        print("Summary DataFrame is empty, skipping plot generation.")
        return

    sns.set_theme(style="whitegrid")
    
    # Sort by a key performance metric for clearer visualization
    summary_df = summary_df.sort_values(by='Avg_Judge_Overall', ascending=False)
    
    metrics_to_plot = {
        'Avg_Cosine_Similarity': 'Average Cosine Similarity',
        'Avg_ROUGE-L': 'Average ROUGE-L Score',
        'Avg_Latency_sec': 'Average Response Latency (sec)',
        'Avg_Judge_Overall': 'Average Judge Score (Overall)'
    }

    for col, title in metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=summary_df[col], y=summary_df.index)
        ax.set_title(title)
        ax.set_xlabel(col.replace('_', ' '))
        ax.set_ylabel('Model')
        # Add labels to the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"summary_{col}.png")
        plt.savefig(plot_path)
        print(f"Saved summary plot to {plot_path}")
        plt.close()

def main(args):
    # --- Setup ---
    configs = load_all_configs(args.configs)
    if not configs:
        print("Error: No valid configuration files were loaded. Exiting.")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Test Data ---
    print(f"\n--- Loading test data from {args.test_file} ---")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Found {len(test_data)} questions for evaluation.")

    # --- Identify Final Models to Evaluate ---
    all_models = os.listdir(args.models_dir)
    final_models = sorted([m for m in all_models if 'self-improve' in m])
    print(f"\n--- Found {len(final_models)} final models to evaluate: ---")
    for m in final_models:
        print(f"  - {m}")
        
    # --- Initialize Evaluation Tools ---
    rouge = evaluate.load('rouge')
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    
    loaded_base_models = {}
    master_summary = []

    # --- Main Evaluation Loop for each model ---
    for model_name in final_models:
        print(f"\n{'='*80}\n--- Evaluating Model: {model_name} ---\n{'='*80}")
        
        # --- START OF CORRECTED LOGIC ---
        try:
            # 1. Parse model name to find base architecture
            base_model_choice = model_name.split('_train_')[-1].replace('_v2', '')
            
            # 2. Find the correct config file by checking its AVAILABLE_MODELS
            config_key_found = None
            for config_name, config_module in configs.items():
                if base_model_choice in config_module.AVAILABLE_MODELS:
                    config_key_found = config_name
                    break # Stop once we find a match

            if not config_key_found:
                raise ValueError(f"No loaded config defines '{base_model_choice}' in its AVAILABLE_MODELS.")
            
            config = configs[config_key_found]
            print(f"  -> Detected base model: '{base_model_choice}'. Using config '{config_key_found}'.")
        except Exception as e:
            print(f"  -> Error parsing model name or finding config for '{model_name}': {e}. Skipping.")
            continue
        # --- END OF CORRECTED LOGIC ---

        # 2. Load Base Model (cached)
        if base_model_choice not in loaded_base_models:
            print(f"  -> Loading base model '{base_model_choice}' for the first time...")
            base_model_id = config.AVAILABLE_MODELS[base_model_choice]
            base_model, tokenizer = load_base_model_and_tokenizer(
                base_model_id, config.BNB_CONFIG, base_model_choice, config.CHAT_TEMPLATES
            )
            loaded_base_models[base_model_choice] = (base_model, tokenizer)
        else:
            print(f"  -> Using cached base model '{base_model_choice}'.")
            base_model, tokenizer = loaded_base_models[base_model_choice]
            
        # 3. Load LoRA Adapter
        adapter_path = os.path.join(args.models_dir, model_name)
        ft_model = load_peft_model(base_model, adapter_path)
        
        model_results = []
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for i, item in enumerate(test_data):
            question = item['question']
            ground_truth = item.get('answer', 'N/A')
            print(f"  -> Processing Question {i+1}/{len(test_data)}...")

            # 4. Generate Response and Measure Latency
            start_time = time.time()
            response = generate_response(ft_model, tokenizer, question, max_new_tokens=1024)
            latency = time.time() - start_time

            # 5. Save Raw Response
            with open(os.path.join(model_output_dir, f"q_{i+1}_response.txt"), "w", encoding='utf-8') as f:
                f.write(response)

            result_row = {
                'Question': question,
                'Ground_Truth': ground_truth,
                'Response': response,
                'Latency_sec': latency,
            }

            # 6. Quantitative Metrics
            if ground_truth != 'N/A':
                rouge_scores = rouge.compute(predictions=[response], references=[ground_truth])
                embeddings = similarity_model.encode([ground_truth, response])
                cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
                result_row.update({
                    'ROUGE-L': rouge_scores['rougeL'],
                    'Cosine_Similarity': cos_sim
                })

            # 7. LLM Judge Scoring
            judge_prompt = _get_llm_judge_scoring_prompt(question, response)
            raw_judgement = _generate_response_gemini(judge_prompt)
            try:
                judgement = json.loads(raw_judgement.strip().replace("```json", "").replace("```", ""))
                result_row.update({
                    'Judge_Accuracy': judgement.get('scores', {}).get('accuracy'),
                    'Judge_Helpfulness': judgement.get('scores', {}).get('helpfulness'),
                    'Judge_Clarity': judgement.get('scores', {}).get('clarity'),
                    'Judge_Justification': judgement.get('justification'),
                    'Judge_Overall': judgement.get('overall_score')
                })
            except Exception as e:
                print(f"    -> Warning: Could not parse judge response. Error: {e}")
            
            model_results.append(result_row)
        
        # 8. Save Detailed Report for this Model
        model_df = pd.DataFrame(model_results)
        report_path = os.path.join(model_output_dir, "detailed_report.csv")
        model_df.to_csv(report_path, index=False)
        print(f"\n  -> ✅ Detailed report for '{model_name}' saved to {report_path}")

        # 9. Append Average Scores to Master Summary
        summary_stats = {
            'Model': model_name,
            'Base_Architecture': base_model_choice,
            'Avg_Latency_sec': model_df['Latency_sec'].mean(),
            'Avg_ROUGE-L': model_df['ROUGE-L'].mean() if 'ROUGE-L' in model_df else 'N/A',
            'Avg_Cosine_Similarity': model_df['Cosine_Similarity'].mean() if 'Cosine_Similarity' in model_df else 'N/A',
            'Avg_Judge_Overall': model_df['Judge_Overall'].mean() if 'Judge_Overall' in model_df else 'N/A'
        }
        master_summary.append(summary_stats)

    # --- Final Summary Generation ---
    print(f"\n{'='*80}\n--- Generating Final Summary Report ---\n{'='*80}")
    if not master_summary:
        print("No models were successfully evaluated. Cannot generate summary report.")
        return
        
    summary_df = pd.DataFrame(master_summary).set_index('Model')
    summary_path = os.path.join(args.output_dir, "_SUMMARY_REPORT.csv")
    summary_df.to_csv(summary_path)
    print(f"🏆 Master summary report saved to {summary_path}")

    generate_summary_plots(summary_df, args.output_dir)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all final, self-improved models.")
    parser.add_argument(
        '--models_dir', 
        type=str, 
        default='trained_models', 
        help="Directory containing all your fine-tuned model folders."
    )
    parser.add_argument(
        '--test_file', 
        type=str, 
        required=True, 
        help="Path to the JSON test file with questions and answers (e.g., model-comp-adv.json)."
    )
    parser.add_argument(
        '--configs', 
        type=str, 
        nargs='+', 
        required=True, 
        help='List of all relevant config files (without .py) e.g., config_GGG config_GMM'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='final_evaluation_reports', 
        help="Directory to save all output reports and plots."
    )

    args = parser.parse_args()
    main(args)