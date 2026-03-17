import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# Import core RAP-LDP modules
from train_rap_ldp import BertModelWithRLDPForward
from rap_ldp_components import (
    SACActor, 
    SharedStateEncoder, 
    build_state_vector, 
    compute_embedding_norm_stats, 
    parse_action, 
    compute_attention_entropy,
    apply_rl_dp_forward
)

logger = logging.getLogger(__name__)

class RAPLDPWrapper(nn.Module):
    """
    Wrapper for RAP-LDP model for attack evaluation.
    Reconstructs the RL environment state during inference to generate
    the dynamic privacy parameters.
    """
    def __init__(self, model_path, device, num_blocks=12, target_epsilon=8.0):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.target_epsilon = target_epsilon
        self.num_blocks = num_blocks
        
        # 1. Load Config and Tokenizer
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.output_attentions = True
        self.config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 2. Load Base Model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        self.model = BertModelWithRLDPForward(self.base_model, num_layers=self.config.num_hidden_layers)
        self.model.to(device)
        self.model.eval()
        
        # 3. Load RL Components
        # State dim: 15 (stats) + num_blocks
        self.state_dim = 15 + num_blocks 
        self.num_layers = self.config.num_hidden_layers + 1
        self.action_dim = self.num_layers + self.num_blocks + 2
        
        self.encoder = SharedStateEncoder(self.state_dim, hidden=128, embed_dim=128).to(device)
        self.actor = SACActor(state_dim=128, action_dim=self.action_dim, hidden=128).to(device)
        
        # Load weights
        rl_policy_path = os.path.join(model_path, "rl_policy.pt")
        if os.path.exists(rl_policy_path):
            checkpoint = torch.load(rl_policy_path, map_location=device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.state_mean = checkpoint['state_norm']['mean'].to(device)
            self.state_var = checkpoint['state_norm']['var'].to(device)
            logger.info(f"Loaded RAP-LDP policy from {rl_policy_path}")
        else:
            logger.warning(f"RL policy not found at {rl_policy_path}! Running with uninitialized policy (random noise).")
            self.state_mean = torch.zeros(self.state_dim).to(device)
            self.state_var = torch.ones(self.state_dim).to(device)

    def normalize_state(self, state):
        return (state - self.state_mean) / torch.sqrt(self.state_var + 1e-8)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            # Handle different model architectures
            if hasattr(self.base_model, 'bert'):
                encoder = self.base_model.bert
            elif hasattr(self.base_model, 'roberta'):
                encoder = self.base_model.roberta
            elif hasattr(self.base_model, 'distilbert'):
                encoder = self.base_model.distilbert
            else:
                encoder = self.base_model
                
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions
            
            # Prepare layer outputs for fusion (Input Embeddings + Encoder Layers)
            layer_outputs = [hidden_states[0]] + list(hidden_states[1:])
            
            # --- RL Decision Phase ---
            current_embeddings = hidden_states[-1]
            norm_stats = compute_embedding_norm_stats(current_embeddings)
            attn_entropy = compute_attention_entropy(attentions)
            
            # Construct State
            # Note: During inference, we assume the budget is available (or test specific budget levels)
            state = build_state_vector(
                embedding_norm_stats=norm_stats,
                utility=0.0,
                spent_eps=self.target_epsilon, # Assume steady state or specific budget point
                batch_loss=0.0,
                num_blocks=self.num_blocks,
                target_epsilon=self.target_epsilon,
                prev_loss=0.0,
                gradient_norm=0.0,
                attention_entropy=attn_entropy,
                device=self.device
            )
            
            norm_state = self.normalize_state(state.unsqueeze(0))
            encoded_state = self.encoder(norm_state)
            action_mu, _ = self.actor(encoded_state)
            action_vec = action_mu.squeeze(0)
            
            # Parse Action
            layer_weights, norm_constants, noise_multiplier, attn_influence = parse_action(
                action_vector=action_vec,
                num_layers=self.num_layers,
                num_blocks=self.num_blocks,
                progress=1.0, 
                eps_ratio=1.0, 
                min_noise_bound=0.4 # Ensure noise is applied during attack eval
            )
            
            # Apply Perturbation (RAP-LDP Core)
            perturbed_embedding = apply_rl_dp_forward(
                layer_outputs=layer_outputs,
                layer_weights=layer_weights,
                norm_constants=norm_constants,
                noise_multiplier=noise_multiplier,
                device=self.device,
                attention_outputs=attentions,
                attn_influence=attn_influence
            )
            
            return perturbed_embedding

# ---------------------------------------------------------------------------- #
# Attack 1: Embedding Inversion (Nearest Neighbor & Semantic Similarity)
# Section V.E.2 of the paper
# ---------------------------------------------------------------------------- #
def run_embedding_inversion(victim_model, tokenizer, device, texts, sample_size=100):
    logger.info("Running Embedding Inversion Attack...")
    
    # Get the public vocabulary embeddings (White-box assumption for Inversion)
    word_embeddings = victim_model.base_model.get_input_embeddings().weight # [Vocab, Hidden]
    
    correct_tokens = 0
    total_tokens = 0
    avg_cosine_sim = 0.0
    
    # Use a subset for evaluation
    sample_texts = texts[:sample_size] 
    
    for text in tqdm(sample_texts, desc="Inverting"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        # 1. Get Perturbed Output (The released data)
        perturbed = victim_model(input_ids, mask) 
        
        # 2. Get Clean Input Embeddings (Ground Truth for Semantic Similarity)
        clean_embeds = victim_model.base_model.get_input_embeddings()(input_ids)
        
        # Metric: Semantic Similarity (Cosine Similarity)
        # Eq. 636 in paper
        cos_sim = F.cosine_similarity(perturbed, clean_embeds, dim=-1) # [1, Seq]
        active_cos = cos_sim * mask
        avg_cosine_sim += active_cos.sum().item()
        
        # Metric: Nearest Neighbor Inversion
        # Eq. 142 / 630 in paper
        seq_len = input_ids.shape[1]
        for i in range(seq_len):
            if mask[0, i] == 0: continue
            
            # Adversary attempts to find v in V that minimizes ||Emb(v) - h_hat||
            # Equivalent to maximizing cosine similarity if norms are handled, 
            # but standard implementation often uses cosine for high-dim spaces.
            vec = perturbed[0, i]
            sim = F.cosine_similarity(vec.unsqueeze(0), word_embeddings, dim=1)
            predicted_id = torch.argmax(sim).item()
            
            if predicted_id == input_ids[0, i].item():
                correct_tokens += 1
            total_tokens += 1
            
    acc = correct_tokens / total_tokens if total_tokens > 0 else 0
    final_sim = avg_cosine_sim / total_tokens if total_tokens > 0 else 0
    
    logger.info(f"Nearest Neighbor Accuracy: {acc:.4f}")
    logger.info(f"Semantic Similarity:       {final_sim:.4f}")
    
    return acc, final_sim

# ---------------------------------------------------------------------------- #
# Attack 2: Membership Inference Attack (MIA)
# Section V.E.1 of the paper
# ---------------------------------------------------------------------------- #
def run_mia(victim_model, tokenizer, device, train_data, test_data, text_key, sample_size=500):
    logger.info("Running Membership Inference Attack (Entropy & Confidence)...")
    
    # Helper to get metrics
    def get_metrics(data, desc):
        entropies = []
        confidences = []
        subset = data[:sample_size]
        
        for item in tqdm(subset, desc=desc):
            text = item[text_key]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            
            # Get perturbed embedding
            perturbed = victim_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            
            # Forward through the classification head (Black-box assumption: Attacker queries model)
            base = victim_model.base_model
            if hasattr(base, 'bert'):
                pooler_out = base.bert.pooler(perturbed)
                logits = base.classifier(pooler_out)
            elif hasattr(base, 'roberta'):
                logits = base.classifier(perturbed)
            elif hasattr(base, 'distilbert'):
                pooler_out = perturbed[:, 0]
                pooler_out = base.pre_classifier(pooler_out)
                pooler_out = nn.ReLU()(pooler_out)
                logits = base.classifier(pooler_out)
            else:
                # Fallback for generic models
                pooler_out = perturbed[:, 0]
                logits = base.classifier(pooler_out)
                
            probs = F.softmax(logits, dim=-1)
            
            # Metric 1: Confidence (Max Probability)
            conf = torch.max(probs).item()
            confidences.append(conf)
            
            # Metric 2: Entropy
            ent = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            entropies.append(ent)
            
        return entropies, confidences

    # 1. Collect metrics for Members (Train) and Non-Members (Test)
    mem_ent, mem_conf = get_metrics(train_data, "MIA (Members)")
    non_ent, non_conf = get_metrics(test_data, "MIA (Non-Members)")
    
    # 2. Calculate Attack Success Rate (ASR) via Threshold Sweep
    def calc_asr(mem_scores, non_scores, reverse=False):
        # Combine labels: 1 for Member, 0 for Non-Member
        all_scores = mem_scores + non_scores
        all_labels = [1]*len(mem_scores) + [0]*len(non_scores)
        
        best_acc = 0.0
        # Sweep thresholds
        thresholds = sorted(all_scores)[::max(1, len(all_scores)//50)]
        
        for t in thresholds:
            if reverse: 
                # For Entropy: Lower is usually Member
                preds = [1 if s < t else 0 for s in all_scores]
            else: 
                # For Confidence: Higher is usually Member
                preds = [1 if s > t else 0 for s in all_scores]
            
            acc = accuracy_score(all_labels, preds)
            if acc > best_acc: best_acc = acc
            
        return best_acc

    # Entropy ASR (Lower entropy -> Member)
    asr_entropy = calc_asr(mem_ent, non_ent, reverse=True)
    
    # Confidence ASR (Higher confidence -> Member)
    asr_confidence = calc_asr(mem_conf, non_conf, reverse=False)
    
    logger.info(f"MIA ASR (Entropy):    {asr_entropy:.4f}")
    logger.info(f"MIA ASR (Confidence): {asr_confidence:.4f}")
    
    return asr_entropy, asr_confidence

def main():
    parser = argparse.ArgumentParser(description="RAP-LDP Privacy Attack Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--task_name", type=str, default="sst2", choices=["sst2", "imdb", "qqp"])
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target epsilon for RL state simulation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)
    
    # Load Data
    logger.info(f"Loading dataset: {args.task_name}")
    if args.task_name == 'imdb':
        train_data = load_dataset('imdb', split='train[:1000]')
        test_data = load_dataset('imdb', split='test[:1000]')
        text_key = 'text'
    elif args.task_name == 'qqp':
        train_data = load_dataset('glue', 'qqp', split='train[:1000]')
        test_data = load_dataset('glue', 'qqp', split='validation[:1000]')
        text_key = 'question1' # QQP has two questions, using q1 for embedding check
    else: # sst2
        train_data = load_dataset('glue', 'sst2', split='train[:1000]')
        test_data = load_dataset('glue', 'sst2', split='validation[:1000]')
        text_key = 'sentence'

    # Load Victim Model
    victim = RAPLDPWrapper(args.model_path, device, target_epsilon=args.epsilon)
    
    print("\n" + "="*60)
    print(f"RAP-LDP Privacy Analysis (Epsilon ~ {args.epsilon})")
    print("="*60)
    
    # Run Attacks
    run_embedding_inversion(victim, victim.tokenizer, device, test_data[text_key])
    run_mia(victim, victim.tokenizer, device, train_data, test_data, text_key)
    
    print("="*60)

if __name__ == "__main__":
    main()
