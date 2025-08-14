from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

model_name_or_path = "tencent/Hunyuan-7B-Instruct"
cache_dir = ""
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", cache_dir = cache_dir)  # You may want to use bfloat16 and/or move to GPU here

def simplify(txt):
    messages = [
        {"role": "user",
          "content": f"""You are a professional image understanding expert, and I am using a large model to generate images. 
                    Please optimize the following prompt words:{txt}. 
                    Make the generated images of higher quality and more realistic. 
                    Please keep the prompt words in English for prompt output."""
         },
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_tensors="pt",
                                                    enable_thinking=True # Toggle thinking mode (default: True)
                                                    )
                                                    
    outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)

    output_text = tokenizer.decode(outputs[0])


    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, output_text, re.DOTALL)

    answer_content = [match.strip() for match in answer_matches][0]
    return answer_content
