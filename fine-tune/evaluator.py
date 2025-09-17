import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from sentence_transformers import SentenceTransformer, util
import random

from model_utils import generate_response
from data_generator import _generate_response_gemini

# --- 1. PLOTTING AND REPORTING UTILS ---

# --- MODIFIED: Added 'experiment_name' to create unique filenames and titles ---
def _plot_loss_curves(log_history, model_version_str, output_dir, experiment_name):
    if not log_history:
        print(f"No log history for model {model_version_str}, skipping loss plot.")
        return
        
    df = pd.DataFrame(log_history)
    
    if 'loss' not in df.columns:
        print(f"No training loss ('loss' key) found in log history for {model_version_str}. Skipping plot.")
        return
        
    train_df = df[df['loss'].notna()]
    
    plt.figure(figsize=(10, 6))
    
    if not train_df.empty:
        plt.plot(train_df['step'], train_df['loss'], label='Training Loss', marker='.')
    
    if 'eval_loss' in df.columns:
        eval_df = df[df['eval_loss'].notna()]
        if not eval_df.empty:
            plt.plot(eval_df['step'], eval_df['eval_loss'], label='Validation Loss', marker='o')
        else:
            print(f"Warning: 'eval_loss' column exists but contains no data for model {model_version_str}.")
    else:
        print(f"Warning: No 'eval_loss' found for model {model_version_str}. Plotting training loss only.")

    # MODIFIED: Add experiment name to the title
    plt.title(f'Loss for Model {model_version_str.upper()} ({experiment_name.upper()} Experiment)')
    plt.xlabel('Training Steps'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    # MODIFIED: Add experiment name to the filename
    filename = os.path.join(output_dir, f'loss_curve_{experiment_name}_{model_version_str}.png')
    plt.savefig(filename); print(f"Loss curve plot saved to {filename}"); plt.close()

# --- MODIFIED: Added 'experiment_name' to create unique filenames and titles ---
def _plot_score_comparison(results_df, output_dir, experiment_name):
    questions = [f"Q{i+1}" for i in range(len(results_df))]
    metrics = ['Cosine Similarity', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    models = ['RAG', 'v1', 'v2']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5 * len(metrics)), sharex=True)
    # MODIFIED: Add experiment name to the title
    fig.suptitle(f'Model Performance Metrics Comparison ({experiment_name.upper()} Experiment)', fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_keys = [f'{metric} ({model})' for model in models]
        scores = results_df[metric_keys].astype(float).values
        x = np.arange(len(questions))
        width = 0.25
        
        rects = []
        for j, model_name in enumerate(models):
            bar_pos = x + (j - 1) * width
            rect = ax.bar(bar_pos, scores[:, j], width, label=f'{model_name}')
            rects.append(rect)

        ax.set_ylabel(metric); ax.set_title(f'Comparison of {metric} Scores')
        ax.set_xticks(x, questions); ax.legend(); ax.set_ylim(0, 1)
        for rect in rects:
            ax.bar_label(rect, padding=3, fmt='%.2f')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # MODIFIED: Add experiment name to the filename
    filename = os.path.join(output_dir, f'quantitative_metrics_comparison_{experiment_name}.png')
    plt.savefig(filename); print(f"Quantitative metrics graph saved to {filename}"); plt.close()


def _format_rag_prompt(original_query, context_str):
    return f"""You are an expert researcher. Use ONLY the provided context to answer the question.
**Original Question:**\n"{original_query}"
**Context:**\n{context_str}
**Synthesized Answer:**"""


# --- 2. TRADITIONAL (QUANTITATIVE) EVALUATION ---

# --- MODIFIED: Added 'experiment_name' to pass down to plotting functions ---
def run_traditional_evaluation(config, model_v1, model_v2, tokenizer, experiment_name):
    print("\n--- Running Traditional Quantitative Evaluation (ROUGE, Cosine Sim) ---")
    if not os.path.exists(config.EVALUATION_DATA_PATH):
        print(f"Evaluation data file not found at {config.EVALUATION_DATA_PATH}. Skipping.")
        return
        
    with open(config.EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        
    rouge = evaluate.load('rouge')
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    
    results = []
    for i, item in enumerate(eval_data):
        q, ground_truth, rag_response = item['question'], item['answer'], item['RAG']
        prompt = _format_rag_prompt(q, "No external context provided. Rely on your fine-tuned knowledge.")
        print(f"  Evaluating Question {i+1}/{len(eval_data)}...")

        v1_response = generate_response(model_v1, tokenizer, prompt)
        v2_response = generate_response(model_v2, tokenizer, prompt)

        with open(os.path.join(config.OUTPUT_DIR_EVALUATION, f"q{i+1}_{experiment_name}_model_v1_response.txt"), "w") as f: f.write(v1_response)
        with open(os.path.join(config.OUTPUT_DIR_EVALUATION, f"q{i+1}_{experiment_name}_model_v2_response.txt"), "w") as f: f.write(v2_response)

        score_rag = rouge.compute(predictions=[rag_response], references=[ground_truth])
        score_v1 = rouge.compute(predictions=[v1_response], references=[ground_truth])
        score_v2 = rouge.compute(predictions=[v2_response], references=[ground_truth])
        
        embeddings = similarity_model.encode([ground_truth, rag_response, v1_response, v2_response])
        cos_sim_rag = util.cos_sim(embeddings[0], embeddings[1]).item()
        cos_sim_v1 = util.cos_sim(embeddings[0], embeddings[2]).item()
        cos_sim_v2 = util.cos_sim(embeddings[0], embeddings[3]).item()

        results.append({
            "Question": q,
            "RAG Response": rag_response, "Cosine Similarity (RAG)": f"{cos_sim_rag:.4f}",
            "ROUGE-1 (RAG)": f"{score_rag['rouge1']:.4f}", "ROUGE-2 (RAG)": f"{score_rag['rouge2']:.4f}", "ROUGE-L (RAG)": f"{score_rag['rougeL']:.4f}",
            "v1 Response": v1_response, "Cosine Similarity (v1)": f"{cos_sim_v1:.4f}",
            "ROUGE-1 (v1)": f"{score_v1['rouge1']:.4f}", "ROUGE-2 (v1)": f"{score_v1['rouge2']:.4f}", "ROUGE-L (v1)": f"{score_v1['rougeL']:.4f}",
            "v2 Response": v2_response, "Cosine Similarity (v2)": f"{cos_sim_v2:.4f}",
            "ROUGE-1 (v2)": f"{score_v2['rouge1']:.4f}", "ROUGE-2 (v2)": f"{score_v2['rouge2']:.4f}", "ROUGE-L (v2)": f"{score_v2['rougeL']:.4f}"
        })

    df = pd.DataFrame(results)
    # MODIFIED: Pass the experiment_name to the plotting function
    _plot_score_comparison(df, config.OUTPUT_DIR_VISUALIZATIONS, experiment_name)
    
    # MODIFIED: Make the HTML report filename unique
    report_path = os.path.join(config.OUTPUT_DIR_VISUALIZATIONS, f'quantitative_report_{experiment_name}.html')
    df.to_html(report_path, escape=False)
    print(f"Quantitative evaluation complete. Report and graph saved in '{config.OUTPUT_DIR_VISUALIZATIONS}/'.")

# ... (The rest of the file, LLM Judge and Dataset Eval, is unchanged but included for completeness) ...
# --- 3. LLM-AS-A-JUDGE (QUALITATIVE) EVALUATION ---
def _get_llm_judge_prompt(question, response_v1, response_v2):
    return f"""You are a meticulous and impartial evaluator for a scientific research context, specifically Carbon Nanotubes (CNTs). Your task is to compare two responses to a given question.
**Evaluation Criteria (Score 1-10):**
1.  **Helpfulness:** How well does the response directly and comprehensively answer the question? (1=Not helpful, 10=Extremely helpful)
2.  **Grounding:** Is the answer factual, non-speculative, and sounds like it's derived from a knowledge base (even if one wasn't provided)? Avoids hallucination. (1=Pure speculation, 10=Perfectly grounded)
3.  **Persona:** Does the response adopt the persona of an expert CNT researcher? Is the language precise, technical, and confident? (1=Generic chatbot, 10=Authentic expert)
**Question:**\n{question}\n**Response V1:**\n{response_v1}\n**Response V2:**\n{response_v2}
**Your Task:**
Provide a score (1-10) for each model on each criterion. Then, provide a brief justification for your scores. Finally, declare a "Winner" (V1, V2, or Tie). Output ONLY a JSON object with the following structure:
{{
  "v1_scores": {{"helpfulness": <score_v1_helpfulness>, "grounding": <score_v1_grounding>, "persona": <score_v1_persona>}},
  "v2_scores": {{"helpfulness": <score_v2_helpfulness>, "grounding": <score_v2_grounding>, "persona": <score_v2_persona>}},
  "justification": "<Your brief analysis here>",
  "winner": "<V1, V2, or Tie>"
}}"""

def run_llm_judge_evaluation(config, model_v1, model_v2, tokenizer):
    print("\n--- Running LLM-as-a-Judge Qualitative Evaluation ---")
    if not os.path.exists(config.EVALUATION_DATA_PATH):
        print(f"Evaluation data file not found at {config.EVALUATION_DATA_PATH}. Skipping.")
        return
    with open(config.EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f: eval_data = json.load(f)
    judge_results = []
    for i, item in enumerate(eval_data):
        print(f"  Judge is evaluating Question {i+1}/{len(eval_data)}...")
        question = item['question']
        prompt = _format_rag_prompt(question, "Rely on your fine-tuned knowledge.")
        response_v1 = generate_response(model_v1, tokenizer, prompt)
        response_v2 = generate_response(model_v2, tokenizer, prompt)
        judge_prompt = _get_llm_judge_prompt(question, response_v1, response_v2)
        raw_judgement = _generate_response_gemini(judge_prompt)
        try:
            judgement_json_str = raw_judgement.strip().replace("```json", "").replace("```", "")
            judgement = json.loads(judgement_json_str)
            flat_result = {"question": question, "winner": judgement.get("winner"), "justification": judgement.get("justification"), "v1_helpfulness": judgement.get("v1_scores", {}).get("helpfulness"), "v1_grounding": judgement.get("v1_scores", {}).get("grounding"), "v1_persona": judgement.get("v1_scores", {}).get("persona"), "v2_helpfulness": judgement.get("v2_scores", {}).get("helpfulness"), "v2_grounding": judgement.get("v2_scores", {}).get("grounding"), "v2_persona": judgement.get("v2_scores", {}).get("persona")}
            judge_results.append(flat_result)
        except Exception as e:
            print(f"    Error parsing judge response for Q{i+1}: {e}")
            print(f"    Raw Response: {raw_judgement}")
    df = pd.DataFrame(judge_results)
    df.to_csv(config.LLM_JUDGE_RESULTS_PATH, index=False)
    print(f"LLM Judge evaluation complete. Results saved to '{config.LLM_JUDGE_RESULTS_PATH}'.")

# --- 4. DATASET "SELF-IMPROVEMENT" EVALUATION ---
def _get_dataset_judge_prompt(qa_v1, qa_v2):
    return f"""You are an evaluator assessing the quality of synthetic training data for a fine-tuning task. Your goal is to determine which of two Question/Answer pairs is superior for training an expert AI assistant.
**Evaluation Criteria (Score 1-10):**
1.  **Depth:** How insightful and non-superficial is the question? Does it require synthesis of information? (1=Trivial, 10=Deeply insightful)
2.  **Specificity:** Is the question technical and specific to the domain, rather than general? (1=Very general, 10=Highly specific)
3.  **Clarity:** Is the Q&A pair well-written, clear, and unambiguous? (1=Confusing, 10=Perfectly clear)
**Q&A Pair from Dataset V1 (Generated by Base Model):**\n- Question: {qa_v1['instruction']}\n- Answer: {qa_v1['output']}\n**Q&A Pair from Dataset V2 (Generated by Fine-Tuned Model):**\n- Question: {qa_v2['instruction']}\n- Answer: {qa_v2['output']}
**Your Task:**
Score each dataset on the criteria above. Provide a brief justification. Declare a "Winner". Output ONLY a JSON object:
{{
  "v1_scores": {{ "depth": <score>, "specificity": <score>, "clarity": <score> }},
  "v2_scores": {{ "depth": <score>, "specificity": <score>, "clarity": <score> }},
  "justification": "<Your brief analysis here>",
  "winner": "<V1, V2, or Tie>"
}}"""

def evaluate_datasets(config):
    print("\n--- Running Dataset Quality Evaluation (Self-Improvement Analysis) ---")
    path_v1 = config.DATASET_V1_COMBINED_PATH
    path_v2 = config.DATASET_V2_COMBINED_PATH
    if not os.path.exists(path_v1) or not os.path.exists(path_v2):
        print("One or both combined dataset files not found. Skipping dataset evaluation.")
        print(f"  - Searched for V1 at: {path_v1}")
        print(f"  - Searched for V2 at: {path_v2}")
        return
    with open(path_v1, 'r') as f: data_v1 = json.load(f)
    with open(path_v2, 'r') as f: data_v2 = json.load(f)
    num_samples_to_judge = min(20, len(data_v1), len(data_v2))
    if num_samples_to_judge == 0:
        print("Datasets are empty. Skipping dataset evaluation.")
        return
    print(f"Judging {num_samples_to_judge} random samples from each combined dataset...")
    samples_v1 = random.sample(data_v1, num_samples_to_judge)
    samples_v2 = random.sample(data_v2, num_samples_to_judge)
    results = []
    for i in range(num_samples_to_judge):
        qa_v1 = samples_v1[i]
        qa_v2 = samples_v2[i]
        judge_prompt = _get_dataset_judge_prompt(qa_v1, qa_v2)
        raw_judgement = _generate_response_gemini(judge_prompt)
        try:
            judgement_json_str = raw_judgement.strip().replace("```json", "").replace("```", "")
            judgement = json.loads(judgement_json_str)
            flat_result = { "winner": judgement.get("winner"), **judgement }
            results.append(flat_result)
        except Exception as e:
            print(f"    Error parsing judge response for sample {i+1}: {e}")
    df = pd.DataFrame(results)
    df.to_csv(config.DATASET_JUDGE_RESULTS_PATH, index=False)
    if not df.empty:
        winner_counts = df['winner'].value_counts()
        print("\nDataset Quality Summary:")
        print(winner_counts)
        print(f"\nResults saved to '{config.DATASET_JUDGE_RESULTS_PATH}'.")