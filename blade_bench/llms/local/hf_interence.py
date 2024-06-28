from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import time
from dotenv import load_dotenv


load_dotenv()


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids, scores, **kwargs):
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(
            self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype
        )

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())


class LocalLLM:
    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.generation_config.cache_implementation = "static"
        self.compiled_model = torch.compile(self.model, mode="reduce-overhead")
        self.device = kwargs.get("device", self.get_default_device())

    def get_model_name(self):
        return self.model_id

    def get_default_device(self):
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def generate(self, user_message: str, system_message: Optional[str] = "", **kwargs):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return self.generate_chat(messages, **kwargs)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.5,
        stop_sequences: Optional[List[int]] = None,
    ):

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if stop_sequences is None:
            if "deepseek-ai" in self.model_id:
                terminators = self.tokenizer.eos_token_id
            else:  # llama models
                terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
            outputs = self.compiled_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
        else:
            stopping_criteria = StoppingCriteriaList()
            stop_sequence_ids = self.tokenizer(
                stop_sequences, return_token_type_ids=False, add_special_tokens=False
            )
            for stop_sequence_input_ids in stop_sequence_ids.input_ids:
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids)
                )
            outputs = self.compiled_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                stopping_criteria=stopping_criteria,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )

        response = outputs[0][input_ids.shape[-1] :]
        resp_str = self.tokenizer.decode(response, skip_special_tokens=True)
        if stop_sequences is not None:
            for stop_sequence in stop_sequences:
                resp_str = resp_str.split(stop_sequence)[0]
        return {
            "response": resp_str,
            "usage": {
                "prompt_tokens": input_ids.shape[-1],
                "completion_tokens": response.shape[-1],
                "total_tokens": outputs[0].shape[-1],
            },
        }


if __name__ == "__main__":

    def main():
        model_id = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"  # "meta-llama/Meta-Llama-3-8B-Instruct"
        # cache_dir = osp.join(os.environ["HF_HOME"], "hub")
        llm = LocalLLM(model_id)
        user_message = "Please write me some code to sort a list of integers in Python using the bubble sort algorithm."
        a = time.time()
        response = llm.generate(user_message, stop_sequences=["```python"])
        print(f"Elapsed: {time.time()- a}")
        print(response["response"])

    main()
