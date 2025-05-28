import torch

# for each question q we sample X outputs o_1, o_2, .., o_x from that we get X rewards 
class GRPOTtrainer:
    def __init__(self, model, tokenizer, reward_fn, device):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.optimizer = torch.optim.AdamW(model.parameters, lr=1e-6)
        self.device = device

    def train_step(self, prompt, G_samples = 5):
        all_responses = [] # [G,]

        # for i in range(G_samples):
        # We'll do all model sampling here, I moved it here from solver_pipeline to make it reward fn,
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        print('[INFO] Starting LLM generation')
        with torch.no_grad():
            MAX_TOKENS = 256
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
            # output_ids = output_ids.cpu()
            
            # 2. Decode the response
          
            # generation = response.split(prompt)[-1].strip()


