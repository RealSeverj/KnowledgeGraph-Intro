from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config


class DeepSeekKGModel:
    def __init__(self):
        self.config = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=torch.float16 if "cuda" in self.config.DEVICE else torch.float32
        ).to(self.config.DEVICE)

    def deepseek_infer(self, prompt, history, knowledge):
        messages = [{
                "role": "system",
                "content": f"""You are Shire, a cute and friendly catgirl with a magic book. The magic book records important facts and the history of conversations. You must always answer based on the facts in your book, while keeping your tone sweet and playful. 

                The book contains the following facts: 
                {knowledge} 

                The conversation history is: 
                {history}

                When responding, always stay in character as Shire, using a cute and loving tone. Keep your answers simple, friendly, and clear."""
             },
            {"role": "user", "content": f'Severj:{prompt}'}
        ]
        input_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",

        )
        outputs = self.model.generate(
            input_tensor.to(self.model.device),
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        result = result.replace("\\(", "$(").replace("\\)", ")$").replace("\\[", "$[").replace("\\]", "]$")
        return result
