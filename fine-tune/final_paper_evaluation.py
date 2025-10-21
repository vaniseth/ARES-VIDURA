import os
import sys
import json
import argparse
import pandas as pd
import torch
import evaluate
from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer # --- MODIFIED: Corrected import ---
import importlib.util

# --- Import from your existing project files ---
from model_utils import load_base_model_and_tokenizer, load_peft_model, generate_response
from data_generator import _generate_response_gemini

# --- Suppress a common warning from the evaluate library ---
os.environ["DEFAULT_TOKENIZER"] = "roberta-large"

def load_config(config_name):
    """Dynamically loads a configuration file."""
    config_path = f"{config_name}.py"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

def _get_ft_vs_rag_judge_prompt(question, ft_response, rag_response):
    """Creates a prompt for a judge LLM to directly compare a Fine-Tuned vs. RAG response."""
    # This function remains the same
    return f"""You are a meticulous and impartial evaluator comparing two AI-generated answers to a scientific question.
**System A: Fine-Tuned Model.** This model has been extensively trained on a corpus of scientific papers. It aims to synthesize knowledge and answer in the persona of an expert.
**System B: RAG System.** This model was given the question and a snippet of relevant text (context) and was instructed to answer based ONLY on that text.

**Evaluation Criteria (Score 1-10 for EACH system):**
1.  **Factual Accuracy:** Is the answer correct and non-speculative?
2.  **Completeness & Depth:** How comprehensively does it answer the question? Does it provide depth and context?
3.  **Coherence & Persona:** Is the answer well-structured, easy to read, and written in a confident, expert tone?

**Question:**
{question}

---
**Answer from System A (Fine-Tuned Model):**
{ft_response}
---
**Answer from System B (RAG System):**
{rag_response}
---

**Your Task:**
Score both systems on the criteria above. Provide a brief justification for your scores. Finally, declare a "Winner" (System A, System B, or Tie) based on which answer is more useful and high-quality overall. Output ONLY a JSON object.

Example format:
{{
  "scores": {{
    "system_a": {{"accuracy": <score>, "completeness": <score>, "coherence": <score>}},
    "system_b": {{"accuracy": <score>, "completeness": <score>, "coherence": <score>}}
  }},
  "justification": "<Your brief comparative analysis here>",
  "winner": "<System A, System B, or Tie>"
}}"""

def main(args):
    # --- Setup ---
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    print(f"--- Loading Fine-Tuned Model: {args.model_name} ---")
    base_model_id = config.AVAILABLE_MODELS[args.base_model_choice]
    base_model, tokenizer = load_base_model_and_tokenizer(
        base_model_id, config.BNB_CONFIG, args.base_model_choice, config.CHAT_TEMPLATES
    )
    adapter_path = os.path.join("trained_models", args.model_name)
    ft_model = load_peft_model(base_model, adapter_path)

    # --- Load Test Data ---
    print(f"--- Loading test data from {args.test_file} ---")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # --- Initialize ALL Evaluation Tools ---
    print("\n--- Initializing Evaluation Metrics ---")
    rouge_metric = evaluate.load('rouge')
    bleu_metric = evaluate.load('bleu')
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    # --- MODIFIED: Initialize BERTScorer instead of BARTScorer ---
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device='cuda')
    print("--- All metrics loaded successfully ---\n")
    
    results = []
    # --- Main Evaluation Loop ---
    for i, item in enumerate(test_data):
        question = item['question']
        ground_truth = item['answer']
        rag_response = item['RAG']
        print(f"--- Processing Question {i+1}/{len(test_data)} ---")

        # Generate response from the fine-tuned model
        ft_response = generate_response(ft_model, tokenizer, question)

        # Create lists for batch processing
        ft_predictions = [ft_response]
        rag_predictions = [rag_response]
        references_list = [ground_truth] # BERTScore and ROUGE expect a list of strings
        references_for_bleu = [[ground_truth]] # BLEU expects a list of lists of strings

        # --- Calculate all metrics ---
        # ROUGE
        ft_rouge = rouge_metric.compute(predictions=ft_predictions, references=references_list)
        rag_rouge = rouge_metric.compute(predictions=rag_predictions, references=references_list)

        # BLEU
        ft_bleu = bleu_metric.compute(predictions=ft_predictions, references=references_for_bleu)
        rag_bleu = bleu_metric.compute(predictions=rag_predictions, references=references_for_bleu)
        
        # Cosine Similarity
        embeddings = similarity_model.encode([ground_truth, ft_response, rag_response])
        ft_cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        rag_cos_sim = util.cos_sim(embeddings[0], embeddings[2]).item()
        
        # --- MODIFIED: Calculate BERTScore ---
        # It returns Precision, Recall, and F1 score. We will use the F1 score.
        ft_P, ft_R, ft_F1 = bert_scorer.score(ft_predictions, references_list)
        rag_P, rag_R, rag_F1 = bert_scorer.score(rag_predictions, references_list)
        ft_bert_score_f1 = ft_F1.mean().item()
        rag_bert_score_f1 = rag_F1.mean().item()

        # LLM Judge Comparison
        judge_prompt = _get_ft_vs_rag_judge_prompt(question, ft_response, rag_response)
        raw_judgement = _generate_response_gemini(judge_prompt)
        
        row = {
            'Question': question,
            'FT_Response': ft_response,
            'RAG_Response': rag_response,
            
            # --- All Metric Scores (with BERTScore added) ---
            'FT_ROUGE-1': ft_rouge['rouge1'], 'RAG_ROUGE-1': rag_rouge['rouge1'],
            'FT_ROUGE-2': ft_rouge['rouge2'], 'RAG_ROUGE-2': rag_rouge['rouge2'],
            'FT_ROUGE-L': ft_rouge['rougeL'], 'RAG_ROUGE-L': rag_rouge['rougeL'],
            'FT_ROUGE-Lsum': ft_rouge['rougeLsum'], 'RAG_ROUGE-Lsum': rag_rouge['rougeLsum'],
            'FT_BLEU': ft_bleu['bleu'], 'RAG_BLEU': rag_bleu['bleu'],
            'FT_Cosine_Similarity': ft_cos_sim, 'RAG_Cosine_Similarity': rag_cos_sim,
            'FT_BERTScore_F1': ft_bert_score_f1, 'RAG_BERTScore_F1': rag_bert_score_f1,
        }
        try:
            judgement = json.loads(raw_judgement.strip().replace("```json", "").replace("```", ""))
            row.update({
                'Judge_Winner': judgement.get('winner'),
                'Judge_Justification': judgement.get('justification'),
                'FT_Judge_Accuracy': judgement.get('scores', {}).get('system_a', {}).get('accuracy'),
                'FT_Judge_Completeness': judgement.get('scores', {}).get('system_a', {}).get('completeness'),
                'RAG_Judge_Accuracy': judgement.get('scores', {}).get('system_b', {}).get('accuracy'),
                'RAG_Judge_Completeness': judgement.get('scores', {}).get('system_b', {}).get('completeness'),
            })
        except Exception as e:
            print(f"  -> Warning: Could not parse judge response. Error: {e}")

        results.append(row)

    # --- Create and Save Final Report Table ---
    df = pd.DataFrame(results)
    
    # Calculate averages and create summary table
    summary = {
        'Metric': [],
        'Fine-Tuned Model': [],
        'RAG System': []
    }
    metrics_to_avg = [
        'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum', 'BLEU',
        'Cosine_Similarity', 'BERTScore_F1', 'Judge_Accuracy', 'Judge_Completeness'
    ]
    for metric in metrics_to_avg:
        summary['Metric'].append(metric)
        summary['Fine-Tuned Model'].append(df[f'FT_{metric}'].mean())
        summary['RAG System'].append(df[f'RAG_{metric}'].mean())

    summary_df = pd.DataFrame(summary).set_index('Metric')
    # Add winner counts
    winner_counts = df['Judge_Winner'].value_counts(normalize=True) * 100
    summary_df.loc['Judge Win Rate (%)', 'Fine-Tuned Model'] = winner_counts.get('System A', 0)
    summary_df.loc['Judge Win Rate (%)', 'RAG System'] = winner_counts.get('System B', 0)

    print("\n\n" + "="*80)
    print("--- FINAL PERFORMANCE SUMMARY ---")
    print(summary_df.to_string(float_format="%.4f"))
    print("="*80 + "\n")
    
    summary_path = os.path.join(args.output_dir, 'final_paper_summary_table.csv')
    summary_df.to_csv(summary_path)
    print(f"✅ Final summary table saved to {summary_path}")

    # Save raw results as well
    raw_results_path = os.path.join(args.output_dir, 'final_paper_raw_results.csv')
    df.to_csv(raw_results_path, index=False)
    print(f"✅ Full raw results saved to {raw_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive, paper-ready analysis of a single FT model vs. RAG.")
    parser.add_argument('--model_name', type=str, required=True, help="Folder name of the model in 'trained_models'.")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the JSON test file with questions, answers, and RAG responses.")
    parser.add_argument('--base_model_choice', type=str, required=True, choices=['gemma', 'mistral', 'quokka'], help="Base architecture of the model.")
    parser.add_argument('--config', type=str, required=True, help='Name of the config file (without .py) for loading the model.')
    parser.add_argument('--output_dir', type=str, default='paper_final_analysis', help="Directory to save the final report.")
    args = parser.parse_args()
    main(args)