import torch
from data.prompts import code_o_solver_prompt, instruction_following

# for each question q we sample X outputs o_1, o_2, .., o_x from that we get X rewards 
class GRPOTtrainer:
    def __init__(self, model, tokenizer, reward_fn, device):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        self.device = device
        self.ref_model = model
    # input args and snippet are deduction specific ones
    def train_step(self, prompt, input_args, snippet, G_samples = 5):
        all_responses = [] # [G,]

        # for i in range(G_samples):
        # We'll do all model sampling here, I moved it here from solver_pipeline to make it reward fn,
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        print('[INFO] Starting LLM generation')
        with torch.no_grad():
            MAX_TOKENS = 15
            output_ids = self.model.generate(
                **input_ids,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=G_samples
            )
            # [G_samples, seq_leng]
            # Move output back to CPU for decoding
        output_ids = output_ids.cpu()
        print(f'Computed G samples')
        # 2. Decode the response
        for i in range(G_samples):
            all_responses.append(self.tokenizer.decode(output_ids[i], skip_special_tokens=True))

        
        # 3. Compute rewards for every compeltion:

        rewards = torch.tensor([self.reward_fn(response, prompt, input_args, snippet) for response in all_responses], dtype=torch.float32)
        print(rewards)
        
        # generation = response.split(prompt)[-1].strip()
        mean_reward = torch.mean(rewards)
        std_reward = torch.std(rewards)
        # Standardize rewards using vectorized operations
        advantages = (rewards - mean_reward) / std_reward if std_reward > 0 else torch.zeros_like(rewards)
        print(std_reward)

        pass

snippet = """def f(x: int):
    return x**2"""
input_args = '3'
task_prompt = code_o_solver_prompt.format(snippet=snippet, input_args=input_args)
prompt = instruction_following.format(task_prompt)

from transformers import AutoModelForCausalLM, AutoTokenizer
from solver_pipeline import reward_fn
# # Load model and tokenizer
model_name = "Qwen/Qwen3-4B"
print(f"[INFO] Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train = GRPOTtrainer(model, tokenizer, reward_fn, device)
train.train_step(prompt, input_args, snippet)