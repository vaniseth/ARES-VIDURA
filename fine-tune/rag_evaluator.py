import json
import re

class RAGEvaluator:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        # The prompt templates are now more complex and need to be stored
        # as class attributes or loaded from a separate file.
        # Goal: Did the retriever fetch the right evidence?
        self.retrieval_rubric = """
            You are a Retrieval Judge. Evaluate the retrieved CONTEXT for the USER QUERY.
            Ignore the final answer quality; judge retrieval only.

            Return JSON with fields:
            {
              "relevance@k": 0-5, // Are top-k chunks about the user's need?
              "coverage": 0-5, // Do chunks collectively cover all key aspects?
              "diversity": 0-5, // Non-duplicative, complementary chunks?
              "noise": 0-5, // 5 = no off-topic/boilerplate
              "missing_aspects": ["..."], // What relevant aspects are not retrieved?
              "notes": "1-3 sentence rationale"
            }

            USER QUERY:
            {query}

            RETRIEVED CONTEXT (ordered):
            {chunks}
        """
        # Goal: Did the generator actually use the retrieved evidence (not hallucinate)?
        self.grounding_rubric = """
            You are a Grounding Judge. Compare the MODEL ANSWER to the RETRIEVED CONTEXT for the USER QUERY.
            Score faithfulness to cited content; do not penalize style.

            Return JSON:
            {
              "grounding": 0-5, // Answer statements traceable to context?
              "citation_alignment": 0-5, // Do citations/attributions point to correct chunks?
              "evidence_coverage": 0-5, // Did the answer leverage the most pertinent chunks?
              "hallucination_flags": ["..."], // specific claims lacking support
              "unused_top_chunks": ["chunk_id", ...], // high-relevance chunks ignored
              "notes": "1-3 sentence rationale"
            }

            USER QUERY: {query}
            MODEL ANSWER: {answer}
            RETRIEVED CONTEXT: {chunks}
        """

        # Goal: Did the system’s intermediate reasoning improve retrieval? (helpful as we are doing query rewriting / multi-hop)
        self.process_rubric = """
            You are a Process Judge. Inspect intermediate steps (rewrites, subqueries, hops).
            Judge whether these steps improved retrieval quality and stayed on-intent.

            Return JSON:
            {
              "intent_preservation": 0-5, // Reformulations keep user's intent?
              "helpfulness_gain": 0-5, // Would these steps likely improve recall/coverage?
              "drift_risk": 0-5, // 5 = minimal drift risk
              "redundancy": 0-5, // 5 = minimal redundancy
              "bad_steps": ["..."], // steps that likely harmed retrieval
              "notes": "1-3 sentence rationale"
            }

            USER QUERY: {query}
            INTERMEDIATE STEPS: {steps}
            RETRIEVED CONTEXT SUMMARY: {brief_list_of_chunks_or_metrics}
        """

    def _call_judge(self, prompt):
        """Helper method to call the LLM and parse the JSON response."""
        try:
            response = self.gemini_model.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0).replace("```json", "").replace("```", "").strip()
                return json.loads(json_text)
            else:
                return None
        except Exception as e:
            print(f"Error calling judge: {e}")
            return None

    def evaluate_retrieval(self, query, chunks):
        prompt = self.retrieval_rubric.format(query=query, chunks=json.dumps(chunks))
        return self._call_judge(prompt)

    def evaluate_grounding(self, query, answer, chunks):
        prompt = self.grounding_rubric.format(query=query, answer=answer, chunks=json.dumps(chunks))
        return self._call_judge(prompt)

    def evaluate_process(self, query, steps, chunks_summary):
        prompt = self.process_rubric.format(query=query, steps=json.dumps(steps), brief_list_of_chunks_or_metrics=json.dumps(chunks_summary))
        return self._call_judge(prompt)



# add this script to the main file:
# from rag_evaluator import RAGEvaluator

# def run_final_evaluation(model_v1, model_v2, tokenizer, eval_data_path):
#     # ... (initial checks) ...
#     evaluator = RAGEvaluator(gemini_model)
#     # ... (retriever setup) ...

#     final_report = []

#     for i, item in enumerate(eval_data):
#         q, ground_truth = item['question'], item['answer']
#         # Simulate retrieval
#         retrieved_contexts = ... # (your retrieval logic from the previous answer)

#         # 1. Evaluate Retrieval with the Judge
#         retrieval_scores = evaluator.evaluate_retrieval(q, retrieved_contexts)

#         # 2. Generate and Evaluate for Model V1
#         rag_prompt_v1 = format_rag_prompt(q, " ".join(retrieved_contexts))
#         v1_response = generate_response_finetuned(model_v1, tokenizer, rag_prompt_v1)
#         grounding_scores_v1 = evaluator.evaluate_grounding(q, v1_response, retrieved_contexts)

#         # 3. Generate and Evaluate for Model V2
#         # (Repeat the generation and grounding evaluation for model v2)
#         v2_response = generate_response_finetuned(model_v2, tokenizer, rag_prompt_v1)
#         grounding_scores_v2 = evaluator.evaluate_grounding(q, v2_response, retrieved_contexts)

#         # 4. Compile the final structured output
#         # You'll need to define the weighted_mean logic.
#         # This is a key part of your rubric that turns the 0-5 scores into a single metric.
#         # Example for Retrieval Health: (relevance*0.4 + coverage*0.3 + diversity*0.2 + noise*0.1) / 5
#         def weighted_mean(scores_dict, weights):
#             total_score = sum(scores_dict[k] * w for k, w in weights.items())
#             total_weight = sum(weights.values())
#             return total_score / total_weight if total_weight > 0 else 0

#         retrieval_weights = {"relevance@k": 0.4, "coverage": 0.3, "diversity": 0.2, "noise": 0.1}
#         retrieval_health = weighted_mean(retrieval_scores, retrieval_weights) if retrieval_scores else 0

#         # Create the final JSON structure for this sample
#         sample_report = {
#             "sample_id": f"question_{i+1}",
#             "scores": {
#                 "retrieval": retrieval_scores,
#                 "grounding_v1": grounding_scores_v1,
#                 "grounding_v2": grounding_scores_v2
#             },
#             "composite": {
#                 "retrieval_health": retrieval_health,
#                 # Add other composite scores here
#             },
#             "flags": {
#                 "hallucinations_v1": grounding_scores_v1.get("hallucination_flags", []),
#                 "hallucinations_v2": grounding_scores_v2.get("hallucination_flags", []),
#                 "missing_aspects": retrieval_scores.get("missing_aspects", [])
#             },
#             "notes": "short merged rationale" # You could have the LLM create this or just skip it
#         }
#         final_report.append(sample_report)

#     # 5. Save the detailed report
#     output_filename = 'evaluation_outputs/detailed_llm_judge_report.json'
#     with open(output_filename, 'w') as f:
#         json.dump(final_report, f, indent=4)
#     print(f"Detailed LLM-as-a-Judge report saved to {output_filename}")

