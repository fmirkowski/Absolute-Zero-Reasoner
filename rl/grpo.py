import torch
from data.prompts import code_o_solver_prompt, instruction_following
import torch.nn.functional as F

# for each question q we sample X outputs o_1, o_2, .., o_x from that we get X rewards 
class GRPOTtrainer:
    def __init__(self, model, tokenizer, reward_fn, device):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        self.device = device
        self.ref_model = model
        self.G_samples = 2
    # input args and snippet are deduction specific ones
    def train_step(self, prompt, input_args, snippet):
        all_responses = [] # [G,]

        # for i in range(G_samples):
        # We'll do all model sampling here, I moved it here from solver_pipeline to make it reward fn,
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        print('[INFO] Starting LLM generation')
        with torch.no_grad():
            MAX_TOKENS = 32
            output_ids = self.model.generate(
                **input_ids,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=self.G_samples,
                output_scores=True,
                return_dict_in_generate=True
            )
        # all_gen_logits_single = torch.tensor([])
        # all_gen_logits = torch.tensor([])
        logits = torch.stack(output_ids.scores, dim=1)  # Shape: [G_samples, max_tokens, vocab_size]
        # more parallerlizable version:
        prompt_length = input_ids.input_ids.shape[1]  # Get length of input prompt
        sequences = output_ids.sequences[:, prompt_length:]
        generated = sequences.copy()
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (sequences != self.tokenizer.pad_token_id).float()
        
        all_gen_logits = torch.gather(logits, dim=-1, index=sequences.unsqueeze(-1)).squeeze(-1)
        # Apply mask to exclude pad tokens
        all_gen_logits = all_gen_logits * attention_mask
        print('\n\n', torch.softmax(all_gen_logits[0], dim=-1), '\n\n', torch.softmax(all_gen_logits[1], dim=-1), '\n\n')
        log_probs = F.log_softmax(all_gen_logits, dim=-1)

        new_log_probs = self.forward_get_log_probs(self.model, input_ids.input_ids, generated)

        # Move output back to CPU for decoding
        output_ids.sequences = output_ids.sequences.cpu()
        print(f'Computed G samples')
        # 2. Decode the response
        for i in range(self.G_samples):
            all_responses.append(self.tokenizer.decode(output_ids.sequences[i], skip_special_tokens=True))

        
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

    def forward_get_log_probs(self, model, input, output_gen):
        # input shape: [G, M]
        for i in range(self.G_samples):
            input_tensor = torch.cat((input[i,:], output_gen[i,:]), dim=-1) # 1D
            attn = torch.ones_like(input_tensor) 
            logits = model(input_tensor, attention_mask=attn) 
            input_length = input.shape[0]
            gen_logits = logits.logits[input_length:-1]
            log_probs = F.log_softmax(gen_logits, dim=-1) # shape: []
            log_probs = torch.gather(log_probs, dim=-1, index=output_gen[i,:].unsqueeze(-1))
        return log_probs
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