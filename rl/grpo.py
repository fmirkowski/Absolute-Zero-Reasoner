import torch

# for each question q we sample X outputs o_1, o_2, .., o_x from that we get X rewards 
class GRPOTtrainer:
    def __init__(self, model, tokenizer, reward_fn):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.optimizer = torch.optim.AdamW(model.parameters, lr=1e-6)

    def train_step(self, prompt, G_samples = 5):
        all_responses = [] # [G,]

        for i in range(G_samples):
            # We'll do all model sampling here, I moved it here from solver_pipeline to make it reward fn,
            #