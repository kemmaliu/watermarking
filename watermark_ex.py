import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility and deterministic text generation
torch.manual_seed(42)
np.random.seed(42)

# --- EXPERIMENTAL CONFIGURATION ---
VOCAB_SIZE = 40        # Vocabulary size mapped to real English words
HIDDEN_DIM = 64
PROMPT_LEN = 3
GENERATION_STEPS = 6   # Length of the generated real-world sentence extension
TRAIN_STEPS = 150       # Live optimization updates per student
SECRET_KEY_SEED = 2026

# --- REAL ENGLISH VOCABULARY MAPPING ---
vocabulary = [
    "The", "AI", "model", "generates", "writes", "creates", "beautiful", 
    "intelligent", "structured", "sentences", "text", "code", "with", 
    "without", "hidden", "visible", "watermarks", "signals", "today", 
    "easily", "perfectly", "and", "but", "accurately", "deeply", "learning",
    "systems", "can", "will", "safely", "protect", "human", "authors", 
    "from", "plagiarism", "cheating", "detection", "the", "a", "of"
]
id_to_word = {i: vocabulary[i] if i < len(vocabulary) else f"word_{i}" for i in range(VOCAB_SIZE)}

# --- TOY LLM ARCHITECTURE ---
class ToyLLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(ToyLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.rnn(embeds)
        logits = self.fc(out[:, -1, :]) # Extract terminal step context
        return logits


# --- CUSTOM WATERMARK DISTILLATION LOSS FUNCTION ---
class WatermarkPreservingLoss(nn.Module):
    def __init__(self, watermark_type='gumbel'):
        super(WatermarkPreservingLoss, self).__init__()
        self.watermark_type = watermark_type
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_probs, target_tokens, alpha, **kwargs):
        loss_ce = self.ce(student_logits, target_tokens)
        if alpha == 0:
            return loss_ce
            
        student_probs = torch.softmax(student_logits, dim=-1)
        
        if self.watermark_type == 'gumbel':
            zeta_t = kwargs.get('zeta_t')
            loss_stat = -torch.mean(torch.sum(student_probs * torch.log(zeta_t + 1e-10), dim=-1))
        elif self.watermark_type == 'inverse':
            log_student_probs = torch.log_softmax(student_logits, dim=-1)
            loss_stat = self.kl(log_student_probs, teacher_probs)
        else:
            loss_stat = torch.tensor(0.0)

        return ((1.0 - alpha) * loss_ce) + (alpha * loss_stat)


# --- AUTOREGRESSIVE REAL-TEXT GENERATION PIPELINE ---
def generate_autoregressive(model, prompt_tokens, steps, watermark_type='gumbel', key_seed=42, watermarked=True):
    model.eval()
    generated_sequence = prompt_tokens.clone()
    rng = np.random.default_rng(key_seed)
    captured_key_stats = []
    
    for step in range(steps):
        with torch.no_grad():
            logits = model(generated_sequence)
            probs = torch.softmax(logits, dim=-1)
        
        # --- BASELINE: UNWATERMARKED SAMPLING ---
        if not watermarked:
            # Sample directly from language logits using standard multinomial selection
            next_token_id = torch.multinomial(probs, num_samples=1)[0]
            
            # The verifier checks these tokens blindly against the hidden random sequence keys
            if watermark_type == 'gumbel':
                zeta_t = torch.from_numpy(rng.random((1, VOCAB_SIZE))).float()
                captured_key_stats.append(zeta_t[0, next_token_id.item()].item())
            elif watermark_type == 'inverse':
                u_t = rng.random()
                pi_t = torch.from_numpy(rng.permutation(VOCAB_SIZE))
                idx = torch.where(pi_t == next_token_id.item())[0][0]
                captured_key_stats.append(abs(u_t - (idx.item() / VOCAB_SIZE)))
                
            next_token = next_token_id

        # --- WATERMARKED SAMPLING ---
        else:
            if watermark_type == 'gumbel':
                zeta_t = torch.from_numpy(rng.random((1, VOCAB_SIZE))).float()
                epsilon = 1e-10
                gumbel_noise = torch.log(zeta_t + epsilon) / (probs + epsilon)
                next_token = torch.argmax(gumbel_noise, dim=-1)
                captured_key_stats.append(zeta_t[0, next_token.item()].item())
                
            elif watermark_type == 'inverse':
                u_t = rng.random()
                pi_t = torch.from_numpy(rng.permutation(VOCAB_SIZE))
                permuted_probs = probs[0][pi_t]
                cdf = torch.cumsum(permuted_probs, dim=0)
                idx = torch.where(cdf >= u_t)[0][0]
                next_token = pi_t[idx].unsqueeze(0)
                captured_key_stats.append(abs(u_t - (idx.item() / VOCAB_SIZE)))
            
        generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(0)], dim=1)
        
    return generated_sequence[0].tolist(), captured_key_stats


# --- POST-GENERATION WATERMARK VERIFIER ---
def verify_sentence_watermark(captured_stats, watermark_type='gumbel'):
    epsilon = 1e-10
    if watermark_type == 'gumbel':
        score = -sum(np.log(1.0 - u + epsilon) for u in captured_stats)
        threshold = GENERATION_STEPS * 1.5  
    else:
        score = -sum(np.log(diff + epsilon) for diff in captured_stats)
        threshold = GENERATION_STEPS * 2.5 
    return score, (score > threshold)


def decode_tokens_to_sentence(token_list):
    return " ".join([id_to_word.get(t, "???") for t in token_list])


# --- EXECUTION PIPELINE FOR THE TRAINED STUDENTS & UNWATERMARKED CONTROL ---
def run_text_generation_experiment(w_type):
    print(f"\n=========================================================================")
    print(f"TRAINING STUDENTS & EVALUATING REAL-LIFE TEXT FOR: {w_type.upper()}")
    print(f"=========================================================================")
    
    # Initialize the fixed, authoritative Teacher model
    teacher_model = ToyLLM(VOCAB_SIZE, HIDDEN_DIM).eval()
    
    # Real-life scenario prompt: [0, 1, 2] -> "The AI model"
    real_prompt = torch.tensor([[0, 1, 2]])  
    
    # Benchmark Teacher Generation
    t_tok, t_keys = generate_autoregressive(teacher_model, real_prompt, GENERATION_STEPS, w_type, SECRET_KEY_SEED, watermarked=True)
    t_score, t_det = verify_sentence_watermark(t_keys, w_type)
    
    print(f"[PROMPT START]    : {decode_tokens_to_sentence(t_tok[:PROMPT_LEN])}")
    print(f"[TEACHER GEN]     : {decode_tokens_to_sentence(t_tok)}")
    print(f"                  -> Verifier Score: {t_score:.2f} | Watermark Detected: {t_det}")
    print(f"-------------------------------------------------------------------------")

    # NEW: Benchmark Unwatermarked Generation using the same model backbone
    uw_tok, uw_keys = generate_autoregressive(teacher_model, real_prompt, GENERATION_STEPS, w_type, SECRET_KEY_SEED, watermarked=False)
    uw_score, uw_det = verify_sentence_watermark(uw_keys, w_type)
    print(f"[UNWATERMARKED]   : {decode_tokens_to_sentence(uw_tok)}")
    print(f"                  -> Verifier Score: {uw_score:.2f} | Watermark Detected: {uw_det} (Expected: False)")
    print(f"-------------------------------------------------------------------------")

    # Run the experiment over the exact alpha ratios outlined in your thesis idea
    alpha_ratios = [0.0, 0.25, 0.50, 0.75, 1.0]
    
    for alpha in alpha_ratios:
        # Start a completely fresh student model from scratch (untrained weights)
        student_model = ToyLLM(VOCAB_SIZE, HIDDEN_DIM)
        optimizer = optim.Adam(student_model.parameters(), lr=0.01)
        criterion = WatermarkPreservingLoss(watermark_type=w_type)
        
        # Distillation / Training Loop
        student_model.train()
        train_rng = np.random.default_rng(SECRET_KEY_SEED + 7)
        
        for step in range(TRAIN_STEPS):
            optimizer.zero_grad()
            simulated_inputs = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))
            
            with torch.no_grad():
                t_logits = teacher_model(simulated_inputs)
                t_probs = torch.softmax(t_logits, dim=-1)
                
                if w_type == 'gumbel':
                    batch_zeta = torch.from_numpy(train_rng.random((1, VOCAB_SIZE))).float()
                    epsilon = 1e-10
                    gumbel_noise = torch.log(batch_zeta + epsilon) / (t_probs + epsilon)
                    batch_targets = torch.argmax(gumbel_noise, dim=-1)
                else:
                    batch_u = train_rng.random()
                    batch_pi = torch.from_numpy(train_rng.permutation(VOCAB_SIZE))
                    permuted_probs = t_probs[0][batch_pi]
                    cdf = torch.cumsum(permuted_probs, dim=0)
                    idx = torch.where(cdf >= batch_u)[0][0]
                    batch_targets = batch_pi[idx].unsqueeze(0)
                    batch_zeta = None
            
            s_logits = student_model(simulated_inputs)
            loss = criterion(s_logits, t_probs, batch_targets, alpha=alpha, zeta_t=batch_zeta)
            loss.backward()
            optimizer.step()
            
        # --- POST-TRAINING REAL LIFE TEXT GENERATION ---
        s_tok, s_keys = generate_autoregressive(student_model, real_prompt, GENERATION_STEPS, w_type, SECRET_KEY_SEED, watermarked=True)
        s_score, s_det = verify_sentence_watermark(s_keys, w_type)
        
        print(f"[STUDENT α={alpha:0.2f}] : {decode_tokens_to_sentence(s_tok)}")
        print(f"                  -> Verifier Score: {s_score:.2f} | Watermark Detected: {s_det}")

    print(f"=========================================================================\n")

if __name__ == "__main__":
    run_text_generation_experiment('gumbel')
    run_text_generation_experiment('inverse')