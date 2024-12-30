from datasets import concatenate_datasets,Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer, TrainingArguments,DataCollatorForSeq2Seq
from peft import LoraConfig,TaskType,get_peft_model
import torch
import os
import warnings
import setproctitle
warnings.filterwarnings('ignore')

def process_func(example,tokenizer):
    MAX_LEN = 1024
    instructions = tokenizer("\n".join(["Human:"+example["instruction"],example["input"]]).strip()+"\n\n Assistant:",add_special_tokens=False)
    response = tokenizer(example["output"],add_special_tokens=False)
    input_ids = instructions["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instructions["attention_mask"] + response["attention_mask"] + [1]
    labels = ([-100] * len(instructions["input_ids"])) + response["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LEN:
        input_ids = input_ids[:MAX_LEN]
        attention_mask = attention_mask[:MAX_LEN]
        labels = labels[:MAX_LEN]
    return {"input_ids":input_ids,"attention_mask":attention_mask,"labels":labels}
def load_da(path_list):
    for path in path_list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
    datasets = [Dataset.from_json(path) for path in path_list]
    ds = concatenate_datasets(datasets)
    return ds

if __name__ == '__main__':
    setproctitle.setproctitle("trx_llama_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_da(["./data/processed_BS_single.json"])

    tokenizer = AutoTokenizer.from_pretrained("./data/pretrained_models/shakechen/Llama-2-7b-hf")
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenized_ds = ds.map(lambda x: process_func(x, tokenizer), num_proc=8, remove_columns=ds.column_names)
    
    model = AutoModelForCausalLM.from_pretrained(
        "./data/pretrained_models/shakechen/Llama-2-7b-hf",
        low_cpu_mem_usage=True,
        device_map = "auto",
        torch_dtype = torch.half,
        use_cache=False
        )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM
    )
    #半精度训练
    model = get_peft_model(model, lora_config)
    model.to(device)
    # for name, param in model.named_parameters():
    #     print(name, param.dtype)
    model.enable_input_require_grads()
    print(model.print_trainable_parameters())
    
    args = TrainingArguments(
        output_dir="./data/finetuned_models/llama7b-singleBS",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        logging_dir="./data/logs/llama-2-7b-hf-global",
        logging_steps=5,
        num_train_epochs=3,
        save_steps=20,
        save_total_limit=10,
        gradient_checkpointing=True,
        fp16=True,  
        learning_rate=5e-5,  
        warmup_steps=10,  
        weight_decay=0.01,  
        max_grad_norm=0.1  
    )   
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        train_dataset=tokenized_ds
    )
    
    trainer.train()
