from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification

model_checkpoint = "/data/Weights/bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# 加载自定义模型
custom_pipeline = pipeline(
    "text-classification",  # 根据你的任务类型修改
    model="./Finetuning/cls/results/bert-epoch10",
    tokenizer=tokenizer
)

# 使用pipeline进行预测
result = custom_pipeline(["I hate this movie","i love this movie"])

def postprocess(res):
    Res_dict = {"LABEL_1":"positive", "LABEL_0":"negative"}
    return Res_dict[res['label']]


presult = list(map(postprocess, result))

print(presult)
