from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig

prompt = "Write a story about an alpaca that went to the zoo."

peft_model_id = 'coniferlabs/flan-ul2-alpaca-lora'
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map="auto", load_in_8bit=True)
model = PeftModel.from_pretrained(model, peft_model_id, device_map={'': 0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model.eval()

tokenized_text = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=tokenized_text, parameters={"min_length": 10, "max_length": 250})

res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(res[0])
