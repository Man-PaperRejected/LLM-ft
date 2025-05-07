'''
使用bert-base模型, 实现情感二分类peft微调
1. 模型:bert-base-uncased
2. 数据集:imdb

ps: 数据集和模型都已经预先下载了
'''

from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate 
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import pipeline


# load dateset
dataset_name = "/data/Dataset/LLM-dataset/imdb"
raw_datasets = load_dataset(dataset_name)

model_checkpoint = "/data/Weights/bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 定义预处理函数
def preprocess_function(examples):
    # 'text' 根据不同的数据集做调整
    text_column = "text"
    return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=128) 

# 后处理函数
def postprocess(res):
        Res_dict = {"LABEL_1":"positive", "LABEL_0":"negative"}
        return Res_dict[res['label']]

def Train():

    # 对整个数据集应用预处理函数
    # batched=True 可以并行处理，加快速度
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # 移除不再需要的原始文本列，并将 'label' 列重命名为 'labels' (transformers Trainer 期望的)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # 设置数据集格式为 PyTorch tensors (如果使用 TensorFlow 则设置为 "tf")
    tokenized_datasets.set_format("torch")




    num_labels = 2 # positive/negative
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # 序列分类任务
        r=16,                        # LoRA 的秩 (Rank)
        lora_alpha=32,               # LoRA 缩放因子
        lora_dropout=0.1,            # LoRA 层的 Dropout
        target_modules=["query", "value", "key", "dense"], 
        bias="none",                
    )
    model = get_peft_model(model, peft_config)

    # 加载评估指标
    metric_path = "/data/cache/hf/modules/evaluate_modules/metrics/evaluate-metric--accuracy/f887c0aab52c2d38e1f8a215681126379eca617f96c447638f751434e8e65b14/accuracy.py"

    metric = evaluate.load(metric_path) 

    # 定义计算指标的函数，Trainer 在评估时会调用
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir="./finetune/Seq-Cls/results/bert-peft",              
        eval_strategy="epoch",         
        save_strategy="epoch",               
        learning_rate=2e-5,         # 默认使用AdamW
        weight_decay=0.01,           
        per_device_train_batch_size=32,      
        per_device_eval_batch_size=32,       
        num_train_epochs=10,                  
        load_best_model_at_end=True,         
        metric_for_best_model="accuracy", 
        save_total_limit=2,  
        label_names=["labels"]  # 由于包装了peft，最好显示设置
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        
    )

    trainer.train()

def inference(texts: List[str]):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    
    # 合并权重  
    model = PeftModel.from_pretrained(model, "finetune/Seq-Cls/results/bert-peft/checkpoint-3910")
    model= model.merge_and_unload()
    
    custom_pipeline = pipeline(
        "text-classification",  
        model=model,
        tokenizer=tokenizer,
        framework = "pt",
    )
    
        # 使用pipeline进行预测
    result = custom_pipeline(texts)
    presult = list(map(postprocess, result))
    print(presult)
    
if __name__ == "__main__":
    # Train()
    inference(["I hate this movie","i love this movie"])
    inference(["In my opinion, the film's storyline is slightly weak and not brilliant enough, but its characterization and deep concept are its strong points."])