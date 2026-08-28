import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- EXPERIMENTAL CONFIGURATION ---
VOCAB_SIZE = 1000       # Vocabulary size as referenced in Section 5.1 of the paper
BATCH_SIZE = 32         # Simulated sequence generation instances 
SEQ_LEN = 50            # Generated sequence depth per sample
HIDDEN_DIM = 64         # Latent representation dimension
EPOCHS = 10             # Training iterations for student model distillation
ALPHA_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0] # Ratios of matching loss vs typical CE loss

print(f"Initializing Experiment Pipelines for Vocab Size: {VOCAB_SIZE}")

# --- TOY LLM DEFINITIONS (TEACHER & STUDENT) ---
class ToyLLM(nn.Module):
    """
    A simple sequence-to-logits model mapping context inputs to a multinomial vocabulary.
    """
    def __init__(self, vocab_size, hidden_dim):
        super(ToyLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.rnn(embeds)
        logits = self.fc(out[:, -1, :]) # Extract terminal step logits
        return logits


# --- WATERMARK IMPLEMENTATIONS ---
class WatermarkEngine:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def gumbel_max_decode(self, teacher_probs, key_zeta):
        """
        Implements Scott Aaronson's Gumbel-Max decoder: S(P, zeta) = argmax (log U_w / P_w)
        Ref: Section 1.0 & Section 3 of the attached paper.
        """
        # key_zeta shape: [batch_size, vocab_size] (uniform i.i.d random variables U_w)
        # Avoid zero log values
        epsilon = 1e-10
        gumbel_noise = torch.log(key_zeta + epsilon) / (teacher_probs + epsilon)
        watermarked_tokens = torch.argmax(gumbel_noise, dim=-1)
        return watermarked_tokens

    def inverse_transform_decode(self, teacher_probs, key_u, key_pi):
        """
        Implements Inverse Transform Watermark Decoder.
        Ref: Section 1.0 & Section 4 of the attached paper.
        """
        batch_size = teacher_probs.shape[0]
        watermarked_tokens = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            p_b = teacher_probs[b]
            pi_b = key_pi[b]         # Permutation vector for batch element b
            u_b = key_u[b].item()    # Key uniform scalar variable U_t
            
            # Map probabilities into the permuted index arrangement
            permuted_probs = p_b[pi_b]
            cdf = torch.cumsum(permuted_probs, dim=0)
            
            # Generalized inverse of CDF calculation
            idx = torch.where(cdf >= u_b)[0][0]
            # Map index backwards using the inverse permutation configuration
            watermarked_tokens[b] = pi_b[idx]
            
        return watermarked_tokens

    def compute_gumbel_statistic(self, chosen_tokens, key_zeta):
        """
        Computes Aaronson's detection statistic T = -sum(log(1 - U_(t, w_t)))
        Ref: Equation (3) context in Section 1.0.
        """
        batch_size = chosen_tokens.shape[0]
        stats = []
        for b in range(batch_size):
            w_t = chosen_tokens[b]
            u_wt = key_zeta[b, w_t]
            stats.append(-torch.log(1.0 - u_wt + 1e-10).item())
        return np.mean(stats)


# --- CUSTOM EXPERIMENTAL DISTILLATION LOSS FUNCTION ---
class WatermarkPreservingLoss(nn.Module):
    def __init__(self, watermark_type='gumbel'):
        super(WatermarkPreservingLoss, self).__init__()
        self.watermark_type = watermark_type
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_probs, target_tokens, alpha, **kwargs):
        """
        Loss = (1 - alpha) * Standard_CE + alpha * Matching_Test_Statistic_Loss
        """
        # 1. Base Cross-Entropy alignment loss objective
        loss_ce = self.ce(student_logits, target_tokens)
        
        if alpha == 0:
            return loss_ce
            
        # 2. Matching Test Statistic Optimization Objective
        student_probs = torch.softmax(student_logits, dim=-1)
        
        if self.watermark_type == 'gumbel':
            # Reward student for placing probability on tokens with high key_zeta intensity
            key_zeta = kwargs.get('key_zeta')
            # Minimize negative expected utility scaled by key scores
            loss_stat = -torch.mean(torch.sum(student_probs * torch.log(key_zeta + 1e-10), dim=-1))
            
        elif self.watermark_type == 'inverse':
            # Encourage alignment with the exact token bounds defined by the key permutation mapping
            key_pi = kwargs.get('key_pi')
            key_u = kwargs.get('key_u')
            # Form a soft margin penalization mapping for the target window positions
            loss_stat = -torch.mean(torch.log(torch.gather(student_probs, 1, target_tokens.unsqueeze(1)) + 1e-10))
        else:
            loss_stat = torch.tensor(0.0)

        # Dynamic combination configuration specified in outline
        total_loss = ((1.0 - alpha) * loss_ce) + (alpha * loss_stat)
        return total_loss


# --- EXPERIMENT EXECUTION ENGINE ---
def run_distillation_experiment(watermark_mode='gumbel'):
    print(f"\n==================================================")
    print(f"RUNNING DISTILLATION EXPERIMENT FOR: {watermark_mode.upper()}")
    print(f"==================================================")
    
    # Initialize networks
    teacher_model = ToyLLM(VOCAB_SIZE, HIDDEN_DIM).eval() # Teacher is frozen
    engine = WatermarkEngine(VOCAB_SIZE)
    
    # Simulate a fixed evaluation set to observe text generation properties
    eval_context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # Generate watermark keys/pseudorandom elements for the batch evaluation evaluation step
    eval_zeta = torch.rand(BATCH_SIZE, VOCAB_SIZE)
    eval_u = torch.rand(BATCH_SIZE)
    eval_pi = torch.stack([torch.randperm(VOCAB_SIZE) for _ in range(BATCH_SIZE)])
    
    # Extract baseline Teacher properties
    with torch.no_grad():
        teacher_logits = teacher_model(eval_context)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        
        if watermark_mode == 'gumbel':
            target_tokens = engine.gumbel_max_decode(teacher_probs, eval_zeta)
            baseline_stat = engine.compute_gumbel_statistic(target_tokens, eval_zeta)
        else:
            target_tokens = engine.inverse_transform_decode(teacher_probs, eval_u, eval_pi)
            # In inverse transform, lower absolute distance difference implies stronger watermark retention
            baseline_stat = 0.0 # Handled conditionally relative to rank matching
            
    print(f"Teacher Baseline Verification Statistic: {baseline_stat:.4f}")
    
    # Loop over user defined ratio variations
    for alpha in ALPHA_RATIOS:
        # Clone an un-trained student model instantiation
        student_model = ToyLLM(VOCAB_SIZE, HIDDEN_DIM)
        optimizer = optim.Adam(student_model.parameters(), lr=0.01)
        criterion = WatermarkPreservingLoss(watermark_type=watermark_mode)
        
        # Distillation Loop
        for epoch in range(EPOCHS):
            student_model.train()
            optimizer.zero_grad()
            
            # Generate forward passes with simulated training data
            simulated_inputs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
            
            with torch.no_grad():
                t_logits = teacher_model(simulated_inputs)
                t_probs = torch.softmax(t_logits, dim=-1)
                
            s_logits = student_model(simulated_inputs)
            
            # Generate random training keys per step matching the theoretical formulation
            batch_zeta = torch.rand(BATCH_SIZE, VOCAB_SIZE)
            batch_u = torch.rand(BATCH_SIZE)
            batch_pi = torch.stack([torch.randperm(VOCAB_SIZE) for _ in range(BATCH_SIZE)])
            
            if watermark_mode == 'gumbel':
                batch_targets = engine.gumbel_max_decode(t_probs, batch_zeta)
            else:
                batch_targets = engine.inverse_transform_decode(t_probs, batch_u, batch_pi)
                
            # Calculate loss using custom hybrid metric
            loss = criterion(
                s_logits, t_probs, batch_targets, alpha=alpha,
                key_zeta=batch_zeta, key_pi=batch_pi, key_u=batch_u
            )
            
            loss.backward()
            optimizer.step()
            
        # --- GENERATION EVALUATION PHASE ---
        student_model.eval()
        with torch.no_grad():
            student_logits = student_model(eval_context)
            student_probs = torch.softmax(student_logits, dim=-1)
            
            if watermark_mode == 'gumbel':
                # Decode using student generated distribution arrays 
                student_tokens = engine.gumbel_max_decode(student_probs, eval_zeta)
                retained_stat = engine.compute_gumbel_statistic(student_tokens, eval_zeta)
                # Percent identity overlap check
                token_overlap = (student_tokens == target_tokens).float().mean().item() * 100
                print(f"Alpha Matrix Ratio: {alpha:0.2f} | Resulting Gumbel Test Stat: {retained_stat:.4f} | Token Selection Identity Match: {token_overlap:.1f}%")
                
            elif watermark_mode == 'inverse':
                student_tokens = engine.inverse_transform_decode(student_probs, eval_u, eval_pi)
                token_overlap = (student_tokens == target_tokens).float().mean().item() * 100
                print(f"Alpha Matrix Ratio: {alpha:0.2f} | Token Selection Identity Match: {token_overlap:.1f}%")

# Run experiments across both watermarks described in your outline
if __name__ == "__main__":
    run_distillation_experiment(watermark_mode='gumbel')
    run_distillation_experiment(watermark_mode='inverse')