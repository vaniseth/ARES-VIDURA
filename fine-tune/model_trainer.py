import os
import torch
import shutil
import transformers
import random
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from model_utils import load_base_model_and_tokenizer

def _prepare_data_splits(dataset_list, prefix):
    """Shuffles and splits data into train/eval sets."""
    random.shuffle(dataset_list)
    train_size = int(0.9 * len(dataset_list))
    train_data = Dataset.from_list(dataset_list[:train_size])
    eval_data = Dataset.from_list(dataset_list[train_size:])
    print(f"Split for '{prefix}': {len(train_data)} training samples, {len(eval_data)} evaluation samples.")
    return train_data, eval_data

def run_training(config, dataset_list, model_choice, output_dir):
    """
    Main function to handle the entire training process for a model version.
    """
    print(f"\n--- Starting Training for model at: {output_dir} ---")
    
    base_model_id = config.AVAILABLE_MODELS[model_choice]
    # --- MODIFIED: Pass model_choice and chat_templates to the loader ---
    model, tokenizer = load_base_model_and_tokenizer(
        base_model_id, 
        config.BNB_CONFIG,
        model_choice,
        config.CHAT_TEMPLATES
    )

    # --- MODIFIED: Tokenize function now uses apply_chat_template for flexibility ---
    def tokenize_function(sample):
        user_content = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}"
        assistant_content = f"### Response:\n{sample['output']}"
        
        chat = [
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content.strip()}
        ]
        
        # This will now use the correct template (Llama-2, Mistral, etc.)
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        
        result = tokenizer(text, truncation=True, max_length=2048, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    train_ds, eval_ds = _prepare_data_splits(dataset_list, output_dir.split('/')[-1])
    tokenized_train = train_ds.map(tokenize_function, remove_columns=list(train_ds.features))
    tokenized_eval = eval_ds.map(tokenize_function, remove_columns=list(eval_ds.features))

    # --- PEFT and LoRA setup ---
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    # --- Trainer Setup ---
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=15,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=config.MAX_TRAINING_STEPS,
            learning_rate=config.LEARNING_RATE,
            logging_steps=config.LOGGING_STEPS,
            bf16=True,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            do_eval=True,
            eval_steps=config.EVAL_STEPS,
            save_steps=config.SAVE_STEPS,
            report_to="none",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False
    
    # --- Train ---
    trainer.train()
    
    # --- THIS IS THE CORRECTED SAVING AND COPYING LOGIC ---
    temp_final_path = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(temp_final_path)
    print(f"--- Finished Training. Temporary model saved to: {temp_final_path} ---")

    model_name = os.path.basename(output_dir)
    permanent_save_path = os.path.join(config.LOCAL_MODEL_SAVE_DIR, model_name)

    if os.path.exists(permanent_save_path):
        shutil.rmtree(permanent_save_path)

    shutil.copytree(temp_final_path, permanent_save_path)
    print(f"✅ Final model adapter successfully copied to: {permanent_save_path}")

    return permanent_save_path, trainer.state.log_history