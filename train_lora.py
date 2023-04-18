import os
import csv
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from tqdm import tqdm


def main():
    accelerator = Accelerator()
    
    batch_size = 1
    gradient_accumulation_steps = 8
    max_length = 512
    lr = 1e-4
    num_epochs = 3
    train_data = "./data/train.csv"
    test_data = "./data/test.csv"

    model_name_or_path = "google/flan-ul2"
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto", load_in_8bit=True)
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    accelerator.print(model.print_trainable_parameters())
    dataset = load_dataset(
            'csv', data_files={
                "train": train_data,
                "validation": test_data,
            }, 
            cache_dir="./cache")


    def preprocess_function(examples):
        model_inputs = tokenizer(examples["question"], max_length=max_length, padding=True, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["answer"], max_length=max_length, padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def append_to_csv(file_name, row):
        file_exists = os.path.isfile(file_name)
        with open(file_name, 'a' if file_exists else 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def capture_batch_analytics(epoch, type, step, loss, total_loss, input_ids, labels):
        filename = f"{epoch}_batch_analytics.csv"
        inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        exp_outputs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        append_to_csv(filename, [type, step, loss, total_loss, inputs[0], exp_outputs[0]])
        if len(inputs) > 1 and len(exp_outputs) > 1:
            append_to_csv(filename, [type, step, loss, total_loss, inputs[1], exp_outputs[1]])

    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=16,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Tokenizing dataset...",
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
        
            capture_batch_analytics(epoch, 'train', step, loss.detach().float(), total_loss, batch["input_ids"], batch["labels"])

        model.eval()
        eval_loss = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            capture_batch_analytics(epoch, 'eval', step, loss.detach().float(), eval_loss, batch["input_ids"], batch["labels"])
    
        model.save_pretrained(f"trained_model-{epoch}")


if __name__ == "__main__":
    main()
