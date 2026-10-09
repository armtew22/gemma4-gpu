import time, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

model_id = "google/gemma-4-E2B-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16)

messages = [{"role": "user", "content": [{"type": "text", "text": "Explain what a GPU kernel is in two sentences."}]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

torch.cuda.synchronize(); start = time.time()
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
torch.cuda.synchronize(); elapsed = time.time() - start

new_tokens = out[0][input_len:]
print(processor.decode(new_tokens, skip_special_tokens=True))
print(f"{len(new_tokens)/elapsed:.1f} tokens/sec | peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
