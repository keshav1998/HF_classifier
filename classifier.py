from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import transformers
import torch
import logging
logging.basicConfig(level=logging.INFO)

model = "falcon-7b-instruct"

# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model, local_files_only=True,
#                                           offload_folder="offload_folder",
#                                         torch_dtype=torch.float32,
#                                         trust_remote_code=True,
#                                         device_map="auto",)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model, 
                                             local_files_only=True, 
                                             trust_remote_code=True)

# tokenizer = AutoTokenizer.from_pretrained(model)

logging.info("Model loaded")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

question = """Classify "Cardiac Pathways Corporation" into either of these classes ['Organisation', "Institution', 'Government Body', 'Other']:\n"""
sequences = pipeline(
    question,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
