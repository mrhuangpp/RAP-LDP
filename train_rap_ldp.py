#!/usr/bin/env python3
"""
RAP-LDP Training Script
Training script for Adaptive Differential Privacy Forward Pass using Reinforcement Learning
"""

import argparse
import math
import os
import random
import copy
import json
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

from rap_ldp_components import (
    SharedStateEncoder,
    ReplayBuffer,
    SACActor,
    SACCritic,
    RunningMeanStd,
    compute_embedding_norm_stats,
    build_state_vector,
    parse_action,
    apply_rl_dp_forward,
    compute_reward,
    compute_attention_entropy,
)
from ldp_mechanisms import get_noise_multiplier
from prv_accountant import Accountant, PRVAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism

# Try to import opacus RDPAccountant (not strictly required, prefer prv_accountant)
try:
    from opacus.accountants import RDPAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    # opacus is no longer strictly required as we use prv_accountant


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


# ---------------------------------------------------------------------------- #
# Privacy Accountant
# ---------------------------------------------------------------------------- #

class NumericalPrivacyAccountant:
    """
    Use prv_accountant (based on numerical methods/PLD) for accurate heterogeneous privacy composition.
    Supports dynamic noise multipliers and sampling rates.
    """

    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self.history = []  # List of tuples: (noise_multiplier, sample_rate, num_steps)
        self.total_steps = 0  # Total steps

    def step(self, noise_multiplier: float, sample_rate: float):
        """Record a privacy step"""
        self.total_steps += 1

        # Debug print (every 100 steps)
        if self.total_steps % 100 == 1:
            print(f"[PrivacyDebug] Step {self.total_steps}: nm={noise_multiplier:.4f}, sr={sample_rate:.6f}")

        # If history is not empty, check if we can merge with the last record
        if self.history:
            last_nm, last_sr, last_n = self.history[-1]
            # If noise multiplier and sampling rate are the same (allow minor error), merge
            # Relax merge tolerance to avoid failure due to minor floating point differences
            if abs(last_nm - noise_multiplier) < 1e-4 and abs(last_sr - sample_rate) < 1e-6:
                self.history[-1] = (last_nm, last_sr, last_n + 1)
                return

        # Otherwise create a new record
        self.history.append((noise_multiplier, sample_rate, 1))

    def get_epsilon(self, delta: float = None) -> float:
        """Calculate consumed privacy budget"""
        if delta is None:
            delta = self.delta

        if not self.history:
            if self.total_steps > 10:
                print(f"[EpsilonDebug] History is EMPTY after {self.total_steps} steps!")
            return 0.0

        # Try using prv_accountant
        try:
            prvs = []
            nums = []

            for nm, sr, n in self.history:
                if nm <= 1e-8: nm = 1e-8
                prv = PoissonSubsampledGaussianMechanism(noise_multiplier=nm, sampling_probability=sr)
                prvs.append(prv)
                nums.append(n)

            accountant = PRVAccountant(
                prvs=prvs,
                max_self_compositions=nums,
                eps_error=0.1,
                delta_error=1e-10
            )

            _, eps_estimate, _ = accountant.compute_epsilon(delta=delta, num_self_compositions=nums)

            if self.total_steps % 100 == 1:
                print(f"[EpsilonDebug] Step {self.total_steps}: History len={len(self.history)}, PRV eps={eps_estimate:.4f}")

            if eps_estimate > 0:
        return eps_estimate
        elif self.total_steps > 10:
            print(f"[EpsilonDebug] WARNING: PRV returned 0.0 after {self.total_steps} steps!")
        except Exception as e:
            print(f"[PrivacyWarning] PRVAccountant failed: {e}")

        # Fallback: if PRV fails or returns 0, and opacus is installed, try using RDP
        if OPACUS_AVAILABLE:
            try:
                rdp = RDPAccountant()
                # Reconstruct RDP history
                for nm, sr, n in self.history:
                    # RDPAccountant.step(noise_multiplier, sample_rate)
                    # But opacus usually accumulates step. Here we simulate the accumulation.
                    # For simplicity, we manually calculate historical RDP
                    pass
                # Since the Opacus RDP interface is complex, we use a simplified RDP calculation function
                from opacus.accountants.analysis import rdp as rdp_analysis
                from opacus.accountants.analysis import gmp as gmp_analysis

                # Calculate alphas
                alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                rdp_total = 0.0

                for nm, sr, n in self.history:
                    rdp_step = rdp_analysis.compute_rdp(q=sr, noise_multiplier=nm, steps=n, orders=alphas)
                    rdp_total += rdp_step

                eps, _ = rdp_analysis.get_privacy_spent(orders=alphas, rdp=rdp_total, delta=delta)
                return eps
            except Exception as e:
                print(f"[PrivacyWarning] RDP fallback failed: {e}")

        return 0.0


# ---------------------------------------------------------------------------- #
# Modified BERT Model with RL-DP-Forward
# ---------------------------------------------------------------------------- #

class BertModelWithRLDPForward(nn.Module):
    """
    Wrap BERT model to support RL-DP-Forward
    Collect layer outputs, apply RL-controlled perturbation
    """

    def __init__(self, base_model, num_layers: int = 12):
        super().__init__()
        self.base_model = base_model
        self.num_layers = num_layers
        self.config = base_model.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=True,  # Must be True to collect output from each layer
        return_dict=None,
        labels=None,
        # RAP-LDP Parameters
        rl_layer_weights=None,
        rl_norm_constants=None,
        rl_noise_multiplier=None,
        rl_attn_influence=None,  # New: Attention influence factor
        device=None,
    ):
        """
        Forward pass, apply RAP-LDP

        Args:
            rl_layer_weights: Layer weights [L]，If None, use uniform weights
            rl_norm_constants: Block normalization constants [k]，If None, use default values
            rl_noise_multiplier: Noise multiplier, if None, no noise is added
            rl_attn_influence: Attention influence factor, if None, not used
        """
        from transformers.modeling_outputs import SequenceClassifierOutput, CausalLMOutputWithCrossAttentions

        # ------------------------------------------------------------------
        # Adapt for GPT-2 (Causal LM)
        # ------------------------------------------------------------------
        if hasattr(self.base_model, 'transformer') and hasattr(self.base_model, 'lm_head'):
            transformer = self.base_model.transformer

            # GPT-2 forward
            outputs = transformer(
                input_ids=input_ids,
                past_key_values=None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            # Prepare layer_outputs
            # GPT-2 hidden_states: (initial_embeds, h_0, h_1, ..., h_n)
            layer_outputs = list(hidden_states)

            # Apply RAP-LDP
            if rl_layer_weights is not None and rl_norm_constants is not None and rl_noise_multiplier is not None:
                # ... (Same logic as BERT, ensure detach) ...
                rl_layer_weights = rl_layer_weights.detach().clone().requires_grad_(False)
                rl_norm_constants = rl_norm_constants.detach().clone().requires_grad_(False)

                num_collected_layers = len(layer_outputs)
                if len(rl_layer_weights) != num_collected_layers:
                    # Simple length alignment logic
                    if len(rl_layer_weights) < num_collected_layers:
                        extra = num_collected_layers - len(rl_layer_weights)
                        rl_layer_weights = torch.cat([
                            rl_layer_weights,
                            torch.ones(extra, device=rl_layer_weights.device, requires_grad=False) / extra
                        ])
                    else:
                        rl_layer_weights = rl_layer_weights[:num_collected_layers]
                    rl_layer_weights = F.softmax(rl_layer_weights, dim=-1).detach().clone().requires_grad_(False)

                perturbed_hidden = apply_rl_dp_forward(
                    layer_outputs=layer_outputs,
                    layer_weights=rl_layer_weights,
                    norm_constants=rl_norm_constants,
                    noise_multiplier=rl_noise_multiplier,
                    device=device or next(self.base_model.parameters()).device,
                    attention_outputs=outputs.attentions,
                    attn_influence=rl_attn_influence if rl_attn_influence is not None else 0.0,
                )
            else:
                perturbed_hidden = hidden_states[-1]

            # GPT-2 LM Head
            lm_logits = self.base_model.lm_head(perturbed_hidden)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
            )

        # ------------------------------------------------------------------
        # Original BERT/RoBERTa/DistilBERT logic
        # ------------------------------------------------------------------
        # Call the BERT part of the base model, collect output from each layer
        if hasattr(self.base_model, 'bert'):
            encoder = self.base_model.bert
        elif hasattr(self.base_model, 'roberta'):
            encoder = self.base_model.roberta
        elif hasattr(self.base_model, 'distilbert'):
            encoder = self.base_model.distilbert
        else:
            # Try using base_model directly (e.g., some specific structures)
            encoder = self.base_model

        # Build forward parameter dictionary
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": True,
            "return_dict": True,
        }

        # Check if DistilBERT
        is_distilbert = hasattr(self.base_model, 'distilbert') or \
            (hasattr(self.base_model.config, 'model_type') and self.base_model.config.model_type == 'distilbert')

        # DEBUG PRINTS
        # print(f"DEBUG: Model type check. is_distilbert={is_distilbert}")
        # print(f"DEBUG: base_model type: {type(self.base_model)}")
        # if hasattr(self.base_model, 'config'):
        #     print(f"DEBUG: config.model_type: {getattr(self.base_model.config, 'model_type', 'N/A')}")

        if position_ids is not None:
            if not is_distilbert:
                forward_kwargs["position_ids"] = position_ids

        if token_type_ids is not None:
            if not is_distilbert:
                # Additional check to ensure safety
                if hasattr(encoder, 'embeddings') and hasattr(encoder.embeddings, 'token_type_ids'):
                    forward_kwargs["token_type_ids"] = token_type_ids
                elif hasattr(self.base_model.config, 'type_vocab_size') and self.base_model.config.type_vocab_size > 0:
                    # Compatible with RoBERTa etc.
                    forward_kwargs["token_type_ids"] = token_type_ids

        bert_outputs = encoder(**forward_kwargs)

        # Get output from each layer
        # hidden_states: tuple of [B, L, D] for each layer
        hidden_states = bert_outputs.hidden_states  # tuple of (embedding, layer_0, ..., layer_11)

        # Prepare candidate layer outputs (including embedding layer and all encoder layers)
        layer_outputs = []
        # 1. Embedding layer
        layer_outputs.append(hidden_states[0])
        # 2. Encoder layers (skip embedding, start from layer 1)
        for i in range(1, len(hidden_states)):
            layer_outputs.append(hidden_states[i])

        if rl_layer_weights is not None and rl_norm_constants is not None and rl_noise_multiplier is not None:
            # Ensure tensors are detached from computation graph (prevent backward errors)
            rl_layer_weights = rl_layer_weights.detach().clone().requires_grad_(False)
            rl_norm_constants = rl_norm_constants.detach().clone().requires_grad_(False)

            # Ensure layer count matches
            num_collected_layers = len(layer_outputs)
            if len(rl_layer_weights) != num_collected_layers:
                # Adjust weight count
                if len(rl_layer_weights) < num_collected_layers:
                    # Extend weights (uniform distribution)
                    extra = num_collected_layers - len(rl_layer_weights)
                    rl_layer_weights = torch.cat([
                        rl_layer_weights,
                        torch.ones(extra, device=rl_layer_weights.device, requires_grad=False) / extra
                    ])
                    rl_layer_weights = F.softmax(rl_layer_weights, dim=-1).detach().clone().requires_grad_(False)
                else:
                    # Truncate weights
                    rl_layer_weights = rl_layer_weights[:num_collected_layers]
                    rl_layer_weights = F.softmax(rl_layer_weights, dim=-1).detach().clone().requires_grad_(False)

            # Apply RAP-LDP
            perturbed_hidden = apply_rl_dp_forward(
                layer_outputs=layer_outputs,
                layer_weights=rl_layer_weights,
                norm_constants=rl_norm_constants,
                noise_multiplier=rl_noise_multiplier,
                device=device or next(self.base_model.parameters()).device,
                attention_outputs=bert_outputs.attentions,  # Pass Attention outputs
                attn_influence=rl_attn_influence if rl_attn_influence is not None else 0.0,  # Pass Attention influence factor
            )
        else:
            # Do not use RL, directly use the output of the last layer
            perturbed_hidden = hidden_states[-1]

        # Use perturbed hidden states for classification
        # Adapt to classifier architectures of different models
        if hasattr(self.base_model, 'bert'):
            # BERT: Use pooler to extract and transform [CLS] token
        pooler_output = self.base_model.bert.pooler(perturbed_hidden)
        logits = self.base_model.classifier(pooler_output)
        elif hasattr(self.base_model, 'roberta'):
            # RoBERTa: Classifier extracts [CLS] internally, needs 3D tensor input
            logits = self.base_model.classifier(perturbed_hidden)
        elif hasattr(self.base_model, 'distilbert'):
            # DistilBERT: Manually extract [CLS] and pass through pre_classifier
            pooler_output = perturbed_hidden[:, 0, :]
            pooler_output = self.base_model.pre_classifier(pooler_output)
            pooler_output = nn.ReLU()(pooler_output)
            logits = self.base_model.classifier(pooler_output)
        else:
            # General processing: manually extract [CLS] token
            pooler_output = perturbed_hidden[:, 0, :]
        logits = self.base_model.classifier(pooler_output)

        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')  # Explicitly specify reduction='mean' to ensure a scalar is returned
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            # Ensure loss is a scalar (handle potential numerical issues)
            if loss.dim() > 0:
                loss = loss.mean()

        # Return results
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,  # Keep original hidden states for state construction
            attentions=bert_outputs.attentions,
        )


# ---------------------------------------------------------------------------- #
# Training Loop with RL-DP-Forward
# ---------------------------------------------------------------------------- #

def train_rl_dp_forward(args):
    """Main training loop"""
    use_gen_bleu = (args.task_name in ["e2e_nlg", "dart"]) and (not args.disable_gen_bleu)
    # Initialize WandB
    if args.use_wandb and WANDB_AVAILABLE:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = args.wandb_run_name or f"bert-{args.task_name}_eps{args.epsilon}_{timestamp}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config=vars(args),
            tags=[args.task_name, f"eps{args.epsilon}", "RL-DP-Forward"],
            resume="allow",  # Allow resume, but don't force
            id=None,  # Let wandb generate unique ID
        )
        print(f"WandB initialized: project={args.wandb_project}, run={run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: --use_wandb specified but wandb not installed. Install with: pip install wandb")

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set GPU
    # Method 1: If GPU specified via command line args, set env var
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu_ids}")

    # Method 2: If env var already set, use it
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        num_gpus = len([g for g in gpu_list if g.strip()])
        print(f"Using {num_gpus} GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"GPUs will be visible as cuda:0, cuda:1, ...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Primary device: {device}")
    print(f"Reward mode: {args.reward_mode}")
    use_rl = not args.disable_rl
    print(f"RL enabled: {use_rl}")

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = args.num_labels
    config.output_attentions = True  # [Critical] Enable Attention output
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # GPT-2 pad_token setting
    if "gpt2" in args.model_name:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = tokenizer.eos_token_id

    if "gpt2" in args.model_name:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            config=config,
        )

    model = BertModelWithRLDPForward(base_model, num_layers=config.num_hidden_layers)
    model = model.to(device)

    # Helper function: get actual model (handle DataParallel)
    def get_model():
        if isinstance(model, nn.DataParallel):
            return model.module
        return model

    # Check if multi-GPU is used (via env vars or args)
    use_multi_gpu = False
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        num_gpus = len([g for g in gpu_list if g.strip()])
        use_multi_gpu = num_gpus > 1
    elif args.gpu_ids:
        num_gpus = len(args.gpu_ids.split(','))
        use_multi_gpu = num_gpus > 1

    # If multi-GPU, use DataParallel
    # Note: DataParallel might affect RAP-LDP implementation, prefer single GPU or env vars
    if use_multi_gpu and args.use_data_parallel:
        model = nn.DataParallel(model)
        print("Model wrapped with DataParallel")
        # Note: when using DataParallel, access model via model.module

    # Load dataset
    if args.task_name in ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]:
        dataset = load_dataset("wikitext", args.task_name)
    elif args.task_name == "e2e_nlg":
        dataset = load_dataset("e2e_nlg")
    elif args.task_name == "imdb":
        dataset = load_dataset("imdb")
    elif args.task_name == "dart":
        # Compatibility for older datasets library (skip verification)
        try:
            dataset = load_dataset("dart", ignore_verifications=True)
        except (TypeError, ValueError):
            # Some versions write ignore_verifications as config name in cache
            # Causing missing "default-ignore_verifications=True" cache config
            dataset = load_dataset("dart")
    else:
        dataset = load_dataset("glue", args.task_name)

    train_dataset = dataset["train"]
    if args.task_name == "imdb":
        eval_dataset = dataset["test"]
    else:
        eval_dataset = dataset["validation"]

    # Task to input field mapping (supports single and pair tasks)
    task_to_keys = {
        "sst2": ("sentence", None),
        "imdb": ("text", None),
        "cola": ("sentence", None),
        "qqp": ("question1", "question2"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("sentence1", "sentence2"),
        "mnli-mm": ("sentence1", "sentence2"),
        "wikitext-2-raw-v1": ("text", None),
        "wikitext-103-raw-v1": ("text", None),
        "e2e_nlg": ("meaning_representation", "human_reference"),
        "dart": ("tripleset", "annotations"),
    }
    if args.task_name not in task_to_keys:
        raise ValueError(f"Unsupported task_name={args.task_name}. Supported: {list(task_to_keys.keys())}")
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def tokenize_function(examples):
        # DART processing (Data-to-Text with RDF triples)
        if args.task_name == "dart":
            # DART format:tripleset (list of triples) || annotations (list of texts)
            # Linearize triples and use the first reference text
            triplesets = examples[sentence1_key]
            annotations = examples[sentence2_key]

            texts = []
            prompts = []
            targets = []
            for tripleset, annotation_list in zip(triplesets, annotations):
                # Linearize triples:["subj", "pred", "obj"] -> "subj | pred | obj"
                if isinstance(tripleset, list) and len(tripleset) > 0:
                    triple_strs = [" | ".join(triple) for triple in tripleset]
                    input_str = " , ".join(triple_strs)
                else:
                    input_str = ""

                # Use the first reference text (annotations is a list of lists)
                if isinstance(annotation_list, list) and len(annotation_list) > 0:
                    if isinstance(annotation_list[0], dict) and "text" in annotation_list[0]:
                        output_str = annotation_list[0]["text"]
                    else:
                        output_str = str(annotation_list[0])
                else:
                    output_str = ""

                texts.append(f"{input_str} || {output_str}")
                prompts.append(f"{input_str} ||")
                targets.append(output_str)

            outputs = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,
            )
            prompt_outputs = tokenizer(
                prompts,
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,
            )
            target_outputs = tokenizer(
                targets,
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,
            )
            # Causal LM task:labels = input_ids
            labels = outputs["input_ids"].copy()
            # Set padding token label to -100
            if tokenizer.pad_token_id is not None:
                for i, label in enumerate(labels):
                    padding_mask = [token_id == tokenizer.pad_token_id for token_id in label]
                    labels[i] = [-100 if is_pad else token_id for is_pad, token_id in zip(padding_mask, label)]
            outputs["labels"] = labels
            outputs["prompt_input_ids"] = prompt_outputs["input_ids"]
            outputs["prompt_attention_mask"] = prompt_outputs["attention_mask"]
            outputs["target_input_ids"] = target_outputs["input_ids"]
            return outputs

        # E2E NLG processing (Data-to-Text)
        if args.task_name == "e2e_nlg":
            # Linearization format: meaning_representation || human_reference
            mrs = examples[sentence1_key]
            refs = examples[sentence2_key]
            # Simple linearization strategy
            texts = [f"{mr} || {ref}" for mr, ref in zip(mrs, refs)]
            prompts = [f"{mr} ||" for mr in mrs]
            targets = [ref for ref in refs]

            outputs = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,
            )
            prompt_outputs = tokenizer(
                prompts,
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,
            )
            target_outputs = tokenizer(
                targets,
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,
            )
            # Causal LM task:labels = input_ids
            labels = outputs["input_ids"].copy()
            # Set padding token label to -100，to ignore when calculating loss
            if tokenizer.pad_token_id is not None:
                for i, label in enumerate(labels):
                    # Find padding positions
                    padding_mask = [token_id == tokenizer.pad_token_id for token_id in label]
                    # Set padding positions to -100
                    labels[i] = [-100 if is_pad else token_id for is_pad, token_id in zip(padding_mask, label)]
            outputs["labels"] = labels
            outputs["prompt_input_ids"] = prompt_outputs["input_ids"]
            outputs["prompt_attention_mask"] = prompt_outputs["attention_mask"]
            outputs["target_input_ids"] = target_outputs["input_ids"]
            return outputs

        # GPT-2 / Causal LM task processing
        if args.task_name.startswith("wikitext"):
            outputs = tokenizer(
                examples[sentence1_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,
            )
            # Causal LM task:labels = input_ids
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        if sentence2_key is None:
            # Single sentence task
            return tokenizer(
                examples[sentence1_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,  # Return lists instead of tensors
            )
        else:
            # Sentence pair task
            return tokenizer(
                examples[sentence1_key],
                examples[sentence2_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors=None,  # Return lists instead of tensors
            )

    # Get column names to remove (exclude label, since we need to keep it)
    columns_to_remove = [col for col in train_dataset.column_names if col != "label"]

    # Apply tokenization and remove original text fields
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,  # Remove original columns, but keep label
        load_from_cache_file=False,  # Do not use cache to avoid old data format issues
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
        load_from_cache_file=False,  # Do not use cache
    )

    # Ensure column names are correct (label should already exist, no rename needed)
    print(f"Train dataset columns: {train_dataset.column_names}")
    print(f"Eval dataset columns: {eval_dataset.column_names}")

    # Custom collate function, since we have already padded
    def collate_fn(batch):
        """Custom collate function, convert data to tensors"""
        # Get all keys
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            if key in ["label", "labels"]:
                collated[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
            else:
                collated[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        return collated

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Initialize RL components
    num_layers = config.num_hidden_layers + 1  # +1 for embedding layer
    num_blocks = args.num_blocks
    encoder = None
    sac_buffer = None
    actor = None
    critic = None
    critic_target = None
    actor_opt = None
    critic_opt = None
    state_norm = None

    # State dimension
    # norm_stats(7) + utility(1) + spent_eps(1) + remaining_budget_ratio(1) + eps_ratio(1) +
    # batch_loss(1) + loss_delta(1) + grad_norm(1) + attn_entropy(1) + blocks(num_blocks)
    # Total: 7 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + num_blocks = 15 + num_blocks
    state_dim = 7 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + num_blocks  # 15 + num_blocks

    if use_rl:
        action_dim = num_layers + num_blocks + 1 + 1  # layer_weights + norm_constants + attn_influence + noise_multiplier
        encoder = SharedStateEncoder(state_dim, hidden=128, embed_dim=128).to(device)

    # SAC components
    sac_buffer = ReplayBuffer(
        max_size=args.sac_buffer_size,
        state_dim=encoder.embed_dim,
        action_dim=action_dim,
        device=device,
    )

    actor = SACActor(
        state_dim=encoder.embed_dim,
        action_dim=action_dim,
        hidden=128,
    ).to(device)

    critic = SACCritic(
        state_dim=encoder.embed_dim,
        action_dim=action_dim,
        hidden=128,
    ).to(device)

    critic_target = copy.deepcopy(critic).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_lr)

    # State normalization
    state_norm = RunningMeanStd(state_dim, device=device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler (warmup)
    # Calculate total training steps and warmup steps
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = 500  # Linearly increase from 0 to target lr in first 500 steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Learning rate scheduler initialized: warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")

    # Privacy accountant - using NumericalPrivacyAccountant (prv_accountant)
    # This method is more accurate than RDP, especially for heterogeneous noise
    privacy_accountant = NumericalPrivacyAccountant(delta=args.delta)
    print("Using NumericalPrivacyAccountant (prv_accountant) for accurate heterogeneous privacy accounting")

    sample_rate = args.train_batch_size / len(train_dataset)

    # Initialize training loss queue (for RL Utility calculation)
    recent_train_losses = deque(maxlen=args.rl_interval)

    # Calculate initial noise multiplier based on target epsilon (if not specified)
    if args.no_dp:
        print("=" * 60)
        print("Warning: --no_dp is enabled, NO differential privacy noise will be added!")
        print("This is a control experiment, only for observing model performance without noise.")
        print("=" * 60)
        default_noise_multiplier = 0.0
    elif args.default_noise_multiplier is None:
        print(f"Calculating initial noise multiplier to match target epsilon={args.epsilon}...")
        from dp_noise import get_noise_multiplier
        computed_noise_multiplier = get_noise_multiplier(
            eps=args.epsilon,
            delta=args.delta,
            batch_size=args.train_batch_size,
            dataset_size=len(train_dataset),
            epoch=args.num_epochs,
            local_dp=False,
            noise_type='aGM'
        )
        print(f"Calculated initial noise multiplier: {computed_noise_multiplier:.4f}")
        default_noise_multiplier = computed_noise_multiplier
    else:
        default_noise_multiplier = args.default_noise_multiplier
        print(f"Using specified noise multiplier: {default_noise_multiplier:.4f}")
        print(f"Warning: The specified noise multiplier may not exactly match the target epsilon={args.epsilon}")

    # Training state
    global_step = 0
    prev_utility = None  # Keep this variable for logging, but use EMA for reward calculation
    utility_ema = None  # New: Exponential moving average of Utility as a more stable baseline
    prev_eps = None
    prev_loss = None  # Used to calculate loss change rate

    # Initialize RL action variables (for non-RL steps)
    # Modify initialization strategy: GPT-2 is extremely sensitive to noise, must be strongly locked to the last layer
    layer_weights = torch.zeros(num_layers, device=device)
    layer_weights[-1] = 10.0
    layer_weights[:-1] = -10.0
    # After Softmax: the last layer is almost 1.0, others are negligible

    norm_constants = torch.ones(num_blocks, device=device) * args.default_norm_c
    noise_multiplier = default_noise_multiplier
    attn_influence = 0.0  # Initialize Attention influence factor to 0, gradually explore

    # Evaluation result logging
    eval_results = []
    bleu_metric = None
    bleu_fallback = None
    if use_gen_bleu:
        try:
            import evaluate
            bleu_metric = evaluate.load("sacrebleu")
        except Exception as e:
            try:
                from sacrebleu import corpus_bleu
                bleu_fallback = corpus_bleu
            except Exception as e2:
                print(f"Warning: failed to load BLEU metric: {e}; {e2}")

    # Calculate total steps (for progress calculation)
    total_steps = args.num_epochs * len(train_loader)

    # Epsilon caching mechanism (optimize performance, avoid frequent calls) privacy_accountant.get_epsilon()）
    epsilon_cache = {'step': -1, 'value': 0.0}
    EPSILON_CACHE_INTERVAL = 20  # Update cache every 20 steps

    def get_epsilon_cached(current_step):
        """Cache epsilon calculation results to avoid performance degradation"""
        if args.no_dp:
            return 0.0
        # Recalculate only when step difference exceeds threshold
        if current_step - epsilon_cache['step'] >= EPSILON_CACHE_INTERVAL:
            epsilon_cache['value'] = privacy_accountant.get_epsilon(delta=args.delta)
            epsilon_cache['step'] = current_step
            print(f"[EpsilonCache] Step {current_step}: Updated cache, eps={epsilon_cache['value']:.4f}")
        return epsilon_cache['value']

    # Core training loop
    import time
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in pbar:
            global_step += 1

            # Prepare inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if "label" in batch:
                labels = batch["label"].to(device)
            elif "labels" in batch:
                labels = batch["labels"].to(device)
            else:
                labels = None

            # Make decisions at RL interval steps
            if use_rl and global_step >= args.sac_start_steps and global_step % args.rl_interval == 0:
                # Use the average training loss of recent rl_interval steps as Utility
                if len(recent_train_losses) > 0:
                    avg_train_loss = sum(recent_train_losses) / len(recent_train_losses)
                    utility = -avg_train_loss  # Negative loss as utility
                else:
                    utility = 0.0

                # Get embeddings of current batch (for state construction)
                actual_model = get_model()

                if hasattr(actual_model.base_model, 'transformer'):  # GPT-2
                    transformer_module = actual_model.base_model.transformer
                elif hasattr(actual_model.base_model, 'bert'):
                    transformer_module = actual_model.base_model.bert
                elif hasattr(actual_model.base_model, 'roberta'):
                    transformer_module = actual_model.base_model.roberta
                elif hasattr(actual_model.base_model, 'distilbert'):
                    transformer_module = actual_model.base_model.distilbert
                else:
                    transformer_module = actual_model.base_model

                with torch.no_grad():
                    outputs = transformer_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                    hidden_states = outputs.hidden_states
                    attentions = outputs.attentions
                    current_embeddings = hidden_states[-1]

                current_attn_entropy = compute_attention_entropy(attentions)
                norm_stats = compute_embedding_norm_stats(current_embeddings)
                spent_eps = get_epsilon_cached(global_step)  # Use cached version

                if len(recent_train_losses) > 0:
                    batch_loss = recent_train_losses[-1]
                else:
                    batch_loss = 0.0

                current_grad_norm = grad_norm

                state = build_state_vector(
                    embedding_norm_stats=norm_stats,
                    utility=utility,
                    spent_eps=spent_eps,
                    batch_loss=batch_loss,
                    num_blocks=num_blocks,
                    target_epsilon=args.epsilon,
                    prev_loss=prev_loss,
                    gradient_norm=current_grad_norm,
                    attention_entropy=current_attn_entropy,
                    device=device,
                )

                state_norm.update(state.unsqueeze(0))
                normalized_state = state_norm.normalize(state.unsqueeze(0))
                encoded_state = encoder(normalized_state)

                # SAC action sampling
                action_raw, _ = actor.sample(encoded_state)
                action_vec = action_raw.squeeze(0)

                progress = global_step / total_steps if total_steps > 0 else 0.0
                current_eps_ratio = spent_eps / args.epsilon if args.epsilon > 0 else 0.0
                layer_weights, norm_constants, noise_multiplier, attn_influence = parse_action(
                    action_vector=action_vec,
                    num_layers=num_layers,
                    num_blocks=num_blocks,
                    progress=progress,
                    eps_ratio=current_eps_ratio,
                    min_noise_bound=default_noise_multiplier,
                    ablation_mode=args.ablation_mode,  # Pass ablation mode
                )

                # Store current state and action (for subsequent reward calculation)
                stored_state = encoded_state.squeeze(0).detach().cpu().numpy()
                stored_action = action_vec.detach().cpu().numpy()

            else:
                # Use previous action (or default)
                if global_step < args.sac_start_steps:
                    # Warm-up: forcefully lock to the last layer to avoid GPT-2 crash
                    layer_weights = torch.zeros(num_layers, device=device)
                    layer_weights[-1] = 10.0
                    layer_weights[:-1] = -10.0

                    # [Key Fix] Increase Clipping Norm during Warm-up to retain more signal
                    # But noise_multiplier MUST be kept unchanged (or larger), never reduced!
                    # Reducing noise_multiplier previously caused an instant Epsilon explosion
                    # default_norm_c now default is 4.0, slightly enlarged to 6.0 for Warm-up
                    warmup_norm_c = args.default_norm_c * 1.5
                    norm_constants = torch.ones(num_blocks, device=device) * warmup_norm_c

                    # Keep default noise multiplier (usually > 0.4) to ensure smooth privacy budget consumption
                    noise_multiplier = default_noise_multiplier
                    attn_influence = 0.0
                # else: Use previous parameters and continue running until next rl_interval

            # ------------------------------------------------------------------
            # Action Smoothing - enforced
            # ------------------------------------------------------------------
            # In the transition_steps after RL intervenes, linearly mix RL policy and Warm-up policy
            # Ensure smooth transition from 'Last Layer Only' to 'RL Adaptive' to prevent gradient explosion
            transition_steps = 500
            if use_rl and global_step >= args.sac_start_steps and global_step < args.sac_start_steps + transition_steps:
                blend_alpha = (global_step - args.sac_start_steps) / float(transition_steps)
                # blend_alpha: 0.0 (Start) -> 1.0 (End)

                # 1. Mix noise multiplier
                # RL output noise might be large or small, smoothly transition to RL output
                target_nm = noise_multiplier
                noise_multiplier = blend_alpha * target_nm + (1 - blend_alpha) * default_noise_multiplier

                # Warm-up policy: use only the last layer
                safe_weights = torch.zeros(num_layers, device=device)
                safe_weights[-1] = 10.0  # Logits before Softmax
                safe_weights[:-1] = -10.0
                safe_weights = F.softmax(safe_weights, dim=-1)  # [0, ..., 0, 1]

                # Current layer_weights (might be RL output logits or post-softmax)
                # Note: if layer_weights are logits, need to softmax first
                # In parse_action, layer_weights is already softmaxed
                # In Warm-up, layer_weights are logits (Line 872)
                # Unified processing: ensure layer_weights is a probability distribution
                if isinstance(layer_weights, torch.Tensor):
                    # Simple heuristic: if sum is close to 1, consider as prob; otherwise logits
                    if abs(layer_weights.sum().item() - 1.0) > 0.1:
                        layer_weights = F.softmax(layer_weights, dim=-1)

                layer_weights = blend_alpha * layer_weights + (1 - blend_alpha) * safe_weights

                # 3. Mix normalization constants
                safe_norms = torch.ones(num_blocks, device=device) * (args.default_norm_c * 1.5)
                norm_constants = blend_alpha * norm_constants + (1 - blend_alpha) * safe_norms

                # 4. Mix Attention Influence
                attn_influence = blend_alpha * attn_influence

            # Ensure tensors are detached from computation graph (prevent backward errors)
            # Use detach().clone() to ensure a completely independent tensor copy
            if isinstance(layer_weights, torch.Tensor):
                layer_weights = layer_weights.detach().clone().requires_grad_(False)
            else:
                layer_weights = torch.tensor(layer_weights, device=device, requires_grad=False)

            if isinstance(norm_constants, torch.Tensor):
                norm_constants = norm_constants.detach().clone().requires_grad_(False)
            else:
                norm_constants = torch.tensor(norm_constants, device=device, requires_grad=False)

            # Ensure tensor is on correct device and does not require gradient
            layer_weights = layer_weights.to(device).requires_grad_(False)
            norm_constants = norm_constants.to(device).requires_grad_(False)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                rl_layer_weights=layer_weights,
                rl_norm_constants=norm_constants,
                rl_noise_multiplier=noise_multiplier,
                rl_attn_influence=attn_influence,  # Pass Attention influence factor
                device=device,
            )

            # Get loss and logits
            loss = outputs.loss
            logits = outputs.logits

            # Ensure loss is a scalar (handle DataParallel)
            # DataParallel might return a tensor with multiple GPU losses, need to average
            if loss is not None and isinstance(loss, torch.Tensor):
                if loss.dim() > 0:
                    # If loss is a vector (multiple GPU loss), take average
                    loss = loss.mean()
                # Ensure loss requires gradient (for backprop)
                if not loss.requires_grad:
                    # This shouldn't happen, but for safety, we recalculate loss
                    # Actually, if loss doesn't need grad, computation graph is broken
                    pass  # If loss doesn't need grad, backward() will fail, but this shouldn't happen

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent gradient explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Calculate gradient norm (for semantic-aware sensitivity metric)
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5  # L2 norm

            optimizer.step()
            lr_scheduler.step()  # Update learning rate (warmup)

            # Record training loss (for RL Utility calculation)
            # Note: must be recorded before RL decision, so there is enough data
            current_loss = loss.item()
            recent_train_losses.append(current_loss)

            # Update previous loss (for calculating loss change rate next time)
            # Note: this update should happen after each training step
            prev_loss = current_loss

            # Update privacy accountant (only if DP enabled)
            # Note: since sensitivity is clipped by norm constant, noise multiplier is noise_multiplier
            # No need to multiply by norm constant again (would lead to wrong epsilon)
            # Norm constant is already used to calc noise std during perturbation（C / noise_multiplier）
            # Use opacus RDPAccountant (if import failed, program already exited)
            if not args.no_dp:
                privacy_accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

            # Calculate reward and update SAC (at RL interval steps)
            if use_rl and global_step >= args.sac_start_steps and global_step % args.rl_interval == 0:
                # Calculate reward
                current_utility = utility
                current_eps = get_epsilon_cached(global_step)

                # Initialize delta variable
                delta_u = 0.0
                delta_eps = 0.0

                # Use EMA (Exponential Moving Average) as Utility baseline
                # This significantly reduces Reward fluctuations due to Batch randomness and DP noise
                # Smaller alpha = smoother baseline (less affected by current fluctuations)
                ema_alpha = 0.1

                if utility_ema is None:
                    utility_ema = current_utility

                if prev_eps is not None:
                    # Calculate improvement relative to long-term average, not previous step
                    delta_u = current_utility - utility_ema
                    delta_eps = current_eps - prev_eps

                    # Pass training progress and budget info
                    reward = compute_reward(
                        delta_utility=delta_u,
                        delta_epsilon=delta_eps,
                        current_step=global_step,
                        total_steps=total_steps,
                        target_epsilon=args.epsilon,
                        current_epsilon=current_eps,
                        reward_mode=args.reward_mode,
                        max_steps=total_steps,
                        steps_in_interval=args.rl_interval,  # Pass interval steps
                    )

                    # Update EMA baseline
                    utility_ema = ema_alpha * current_utility + (1 - ema_alpha) * utility_ema
                else:
                    reward = 0.0

                prev_utility = current_utility
                prev_eps = current_eps

                # Log RL decision metrics to WandB
                if args.use_wandb and WANDB_AVAILABLE:
                    # Calculate action stats (monitor if RL is exploring/converging)
                    lw = layer_weights.detach()
                    lw_prob = torch.softmax(lw, dim=0)
                    lw_entropy = float(-(lw_prob * (lw_prob + 1e-8).log()).sum().item())
                    lw_min = float(lw.min().item())
                    lw_max = float(lw.max().item())
                    lw_mean = float(lw.mean().item())

                    nc = norm_constants.detach()
                    nc_mean = float(nc.mean().item())
                    nc_std = float(nc.std().item())
                    nc_min = float(nc.min().item())
                    nc_max = float(nc.max().item())

                    wandb.log({
                        "RL/reward": reward,
                        "RL/delta_utility": delta_u,
                        "RL/delta_epsilon": delta_eps,
                        "RL/utility": current_utility,
                        "RL/attention_entropy": current_attn_entropy,
                        "RL/noise_multiplier": noise_multiplier,
                        "RL/attn_influence": attn_influence,  # Log Attention influence factor
                        "RL/avg_norm_c": float(norm_constants.mean().item()),
                        "RL/lw_entropy": lw_entropy,
                        "RL/lw_min": lw_min,
                        "RL/lw_max": lw_max,
                        "RL/lw_mean": lw_mean,
                        "RL/norm_c_mean": nc_mean,
                        "RL/norm_c_std": nc_std,
                        "RL/norm_c_min": nc_min,
                        "RL/norm_c_max": nc_max,
                    }, step=global_step)

                # Store transition (next_state uses current state)
                sac_buffer.add(
                    stored_state,
                    stored_action,
                    reward,
                    stored_state,  # Simplify: use same state as next_state
                    done=False,
                )

                # SAC update
                if sac_buffer.size >= args.sac_batch_size:
                    for _ in range(args.sac_updates_per_interval):
                        s_b, a_b, r_b, s2_b, d_b = sac_buffer.sample(args.sac_batch_size)

                        # Critic update
                        with torch.no_grad():
                            a2, logp2 = actor.sample(s2_b)
                            q1_t, q2_t = critic_target(s2_b, a2)
                            q_targ = torch.min(q1_t, q2_t) - args.alpha * logp2
                            y = r_b + args.gamma * (1 - d_b) * q_targ

                        q1_c, q2_c = critic(s_b, a_b)
                        loss_q = F.smooth_l1_loss(q1_c, y) + F.smooth_l1_loss(q2_c, y)
                        critic_opt.zero_grad()
                        loss_q.backward()
                        critic_opt.step()

                        # Actor update
                        a_pi, logp_pi = actor.sample(s_b)
                        q1_pi, q2_pi = critic(s_b, a_pi)
                        q_pi = torch.min(q1_pi, q2_pi)
                        loss_pi = (args.alpha * logp_pi - q_pi).mean()
                        actor_opt.zero_grad()
                        loss_pi.backward()
                        actor_opt.step()

                        # Soft-update target critic
                        for p, p_t in zip(critic.parameters(), critic_target.parameters()):
                            p_t.data.mul_(1 - args.tau)
                            p_t.data.add_(args.tau * p.data)

                        # Log RL training metrics to WandB
                        if args.use_wandb and WANDB_AVAILABLE:
                            wandb.log({
                                "RL/actor_loss": loss_pi.item(),
                                "RL/critic_loss": loss_q.item(),
                                "RL/q_value": q_pi.mean().item(),
                            }, step=global_step)

            # Update progress bar
            if args.no_dp:
                current_eps = 0.0
                eps_remaining = 0.0
                eps_ratio = 0.0
            else:
                current_eps = get_epsilon_cached(global_step)
            eps_remaining = max(0.0, args.epsilon - current_eps)
            eps_ratio = current_eps / args.epsilon if args.epsilon > 0 else 0.0

            # Check if target epsilon exceeded (stop only when target reached)
            warn_ratio = 0.99
            if current_eps >= args.epsilon * warn_ratio:
                print(f"\nWarning: Consumed {current_eps:.4f}/{args.epsilon:.4f} epsilon ({eps_ratio * 100:.1f}%)")
                if current_eps >= args.epsilon:
                    print(f"Target epsilon reached, stopping training early")
                    break

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'eps': f"{current_eps:.3f}/{args.epsilon:.3f}",
                'eps%': f"{eps_ratio * 100:.1f}%",
            })

            # Log training metrics to WandB
            if args.use_wandb and WANDB_AVAILABLE:
                # Get current learning rate
                current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else args.learning_rate
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/epsilon": current_eps,
                    "train/epsilon_ratio": eps_ratio,
                    "train/epsilon_remaining": eps_remaining,
                }, step=global_step)

        # Evaluate at the end of each epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs} Performance metrics:")
        print(f"  Duration (Time): {epoch_duration:.2f} s")
        print(f"  Peak Memory: {peak_memory_mb:.2f} MB")
        print(f"{'=' * 60}")

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs} Evaluation results:")
        model.eval()
        eval_losses = []
        eval_preds = []
        eval_labels_list = []
        bleu_predictions = []
        bleu_references = []
        gen_samples_printed = 0

        with torch.no_grad():
            for eval_batch in tqdm(eval_loader, desc="Evaluating"):
                eval_input_ids = eval_batch["input_ids"].to(device)
                eval_attention_mask = eval_batch["attention_mask"].to(device)
                if "label" in eval_batch:
                    eval_labels = eval_batch["label"].to(device)
                elif "labels" in eval_batch:
                    eval_labels = eval_batch["labels"].to(device)
                else:
                    eval_labels = None

                # Evaluate using current RL policy
                outputs = model(
                    input_ids=eval_input_ids,
                    attention_mask=eval_attention_mask,
                    labels=eval_labels,
                    rl_layer_weights=layer_weights.detach().clone().requires_grad_(False).to(device),
                    rl_norm_constants=norm_constants.detach().clone().requires_grad_(False).to(device),
                    rl_noise_multiplier=noise_multiplier,
                    rl_attn_influence=attn_influence,  # Pass Attention influence factor
                    device=device,
                )

                loss = outputs.loss
                logits = outputs.logits

                # Ensure loss is a scalar (handle DataParallel)
                # DataParallel might return a tensor with multiple GPU losses, need to average
                if loss is not None and isinstance(loss, torch.Tensor):
                    if loss.dim() > 0:
                        # If loss is a vector (multiple GPU loss), take average
                        loss = loss.mean()

                eval_losses.append(loss.item())

                # For classification tasks, calculate accuracy
                if eval_labels is not None and logits is not None:
                    predictions = torch.argmax(logits, dim=-1)
                    eval_preds.extend(predictions.cpu().numpy())
                eval_labels_list.extend(eval_labels.cpu().numpy())
                if use_gen_bleu and bleu_metric is not None:
                    if "prompt_input_ids" in eval_batch and "target_input_ids" in eval_batch:
                        prompt_input_ids = eval_batch["prompt_input_ids"].to(device)
                        prompt_attention_mask = eval_batch["prompt_attention_mask"].to(device)
                        target_input_ids = eval_batch["target_input_ids"].to(device)

                        # Truncate to max prompt length in current batch, reduce invalid padding
                        prompt_lengths = prompt_attention_mask.sum(dim=1)
                        prompt_lengths_cpu = prompt_lengths.tolist()
                        max_prompt_len = int(prompt_lengths.max().item())
                        prompt_input_ids = prompt_input_ids[:, :max_prompt_len]
                        prompt_attention_mask = prompt_attention_mask[:, :max_prompt_len]

                        # Greedy decoding (true generative BLEU)
                        gen_input_ids = prompt_input_ids
                        gen_attention_mask = prompt_attention_mask
                        finished = torch.zeros(gen_input_ids.size(0), dtype=torch.bool, device=device)

                        for step_idx in range(args.gen_max_new_tokens):
                            gen_outputs = model(
                                input_ids=gen_input_ids,
                                attention_mask=gen_attention_mask,
                                labels=None,
                                rl_layer_weights=layer_weights.detach().clone().requires_grad_(False).to(device),
                                rl_norm_constants=norm_constants.detach().clone().requires_grad_(False).to(device),
                                rl_noise_multiplier=noise_multiplier,
                                rl_attn_influence=attn_influence,
                                device=device,
                            )
                            next_token = torch.argmax(gen_outputs.logits[:, -1, :], dim=-1)
                            if step_idx + 1 < args.gen_min_new_tokens:
                                next_token = torch.where(
                                    next_token == tokenizer.eos_token_id,
                                    torch.full_like(next_token, tokenizer.pad_token_id),
                                    next_token,
                                )
                            next_token = torch.where(
                                finished, torch.full_like(next_token, tokenizer.eos_token_id), next_token
                            )

                            gen_input_ids = torch.cat([gen_input_ids, next_token.unsqueeze(1)], dim=1)
                            gen_attention_mask = torch.cat(
                                [gen_attention_mask, torch.ones_like(next_token).unsqueeze(1)], dim=1
                            )

                            finished = finished | (next_token == tokenizer.eos_token_id)
                            if finished.all():
                                break

                        generated_texts = []
                        for i in range(gen_input_ids.size(0)):
                            prompt_len = int(prompt_lengths_cpu[i])
                            gen_ids = gen_input_ids[i, prompt_len:]
                            generated_texts.append(
                                tokenizer.decode(gen_ids, skip_special_tokens=True)
                            )
                        bleu_predictions.extend(generated_texts)
                        decoded_refs = tokenizer.batch_decode(target_input_ids, skip_special_tokens=True)
                        bleu_references.extend(decoded_refs)

                        if args.gen_log_samples > 0 and gen_samples_printed < args.gen_log_samples:
                            decoded_prompts = tokenizer.batch_decode(
                                prompt_input_ids, skip_special_tokens=True
                            )
                            for p, g, r in zip(decoded_prompts, generated_texts, decoded_refs):
                                if gen_samples_printed >= args.gen_log_samples:
                                    break
                                print("\n[GenSample]")
                                print(f"PROMPT: {p}")
                                print(f"GEN: {g}")
                                print(f"REF: {r}")
                                gen_samples_printed += 1

        # Calculate evaluation metrics
        eval_loss = np.mean(eval_losses)
        eval_accuracy = None
        bleu_score = None

        # Calculate accuracy (classification tasks)
        if eval_preds and eval_labels_list:
            eval_preds_arr = np.array(eval_preds)
            eval_labels_arr = np.array(eval_labels_list)
            eval_accuracy = (eval_preds_arr == eval_labels_arr).mean()

        # Calculate BLEU (generation tasks)
        if use_gen_bleu and bleu_metric is not None and bleu_predictions and bleu_references:
            bleu_score = bleu_metric.compute(
                predictions=bleu_predictions,
                references=[[ref] for ref in bleu_references],
            )["score"]
        elif use_gen_bleu and bleu_fallback is not None and bleu_predictions and bleu_references:
            bleu_score = bleu_fallback(bleu_predictions, [bleu_references]).score

        current_eps = 0.0 if args.no_dp else privacy_accountant.get_epsilon(delta=args.delta)

        # Print evaluation results
        print(f"  Eval Loss: {eval_loss:.4f}")
        if eval_accuracy is not None:
            print(f"  Accuracy: {eval_accuracy:.4f} ({eval_accuracy * 100:.2f}%)")
        if bleu_score is not None:
            print(f"  BLEU: {bleu_score:.4f}")
        print(f"  Privacy Budget (Epsilon): {current_eps:.4f}")
        print(f"{'=' * 60}\n")

        # Log evaluation metrics to WandB
        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                "eval/loss": eval_loss,
                "eval/epsilon": current_eps,
                "epoch": epoch + 1,
            }
            if eval_accuracy is not None:
                log_dict["eval/accuracy"] = eval_accuracy
            if bleu_score is not None:
                log_dict["eval/bleu"] = bleu_score
            wandb.log(log_dict, step=global_step)

        # Save evaluation results
        result_entry = {
            'epoch': epoch + 1,
            'eval_loss': eval_loss,
            'epsilon': current_eps,
        }
        if eval_accuracy is not None:
            result_entry['accuracy'] = eval_accuracy
        if bleu_score is not None:
            result_entry['bleu'] = bleu_score
        eval_results.append(result_entry)

        # Save RL policy snapshot after each epoch (for analysis)
        if use_rl:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            epoch_rl_stats = {
                "epoch": epoch + 1,
                "layer_weights": layer_weights.detach().cpu().tolist(),
                "norm_constants": norm_constants.detach().cpu().tolist(),
                "noise_multiplier": float(noise_multiplier),
                "attn_influence": float(attn_influence),
                "eval_loss": eval_loss,
                "epsilon": current_eps,
            }
            if eval_accuracy is not None:
                epoch_rl_stats["accuracy"] = eval_accuracy
            if bleu_score is not None:
                epoch_rl_stats["bleu"] = bleu_score

            # Save as epoch-specific file
            epoch_stats_file = output_dir / f"rl_policy_stats_epoch{epoch + 1}.json"
            with open(epoch_stats_file, 'w', encoding='utf-8') as f:
                json.dump(epoch_rl_stats, f, indent=2)
            print(f"  RL policy snapshot saved: {epoch_stats_file}")

        model.train()

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use get_model() to ensure correct saving with DataParallel
    actual_model = get_model()
    actual_model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save RL policy stats (for ablation analysis)
    if use_rl:
        rl_stats = {
            "layer_weights": layer_weights.detach().cpu().tolist(),
            "norm_constants": norm_constants.detach().cpu().tolist(),
            "noise_multiplier": float(noise_multiplier),
            "attn_influence": float(attn_influence)
        }
        with open(output_dir / "rl_policy_stats.json", 'w', encoding='utf-8') as f:
            json.dump(rl_stats, f, indent=2)
        print(f"RL policy stats saved to: {output_dir / 'rl_policy_stats.json'}")

    # Save RL policy
    if use_rl:
        torch.save({
            'actor': actor.state_dict(),
            'critic': critic.state_dict(),
            'critic_target': critic_target.state_dict(),
            'encoder': encoder.state_dict(),
            'state_norm': {
                'mean': state_norm.mean.cpu(),
                'var': state_norm.var.cpu(),
                'count': state_norm.count,
            },
        }, output_dir / "rl_policy.pt")

    eval_results_file = output_dir / "eval_results.json"
    with open(eval_results_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("Training complete! Evaluation result summary:")
    print(f"{'=' * 60}")
    for result in eval_results:
        summary = f"Epoch {result['epoch']}: Loss={result['eval_loss']:.4f}"
        if 'accuracy' in result:
            summary += f", Accuracy={result['accuracy']:.4f} ({result['accuracy'] * 100:.2f}%)"
        if 'bleu' in result:
            summary += f", BLEU={result['bleu']:.4f}"
        summary += f", Epsilon={result['epsilon']:.4f}"
        print(summary)
    print(f"{'=' * 60}")
    print(f"Model saved to: {output_dir}")
    print(f"Evaluation results saved to: {eval_results_file}")

    # End WandB run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("WandB run completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAP-LDP Training")

    # Model and Data
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=128)

    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation_mode", type=str, default="none", choices=["none", "no_alf", "no_sana"],
                        help="Ablation mode: none=full method, no_alf=disable Adaptive Layer Fusion, no_sana=disable Semantics-Aware Noise Allocation")

    # RL Parameters
    parser.add_argument("--num_blocks", type=int, default=12, help="Number of embedding blocks")
    parser.add_argument("--rl_interval", type=int, default=50, help="RL decision interval (steps)")
    parser.add_argument("--sac_start_steps", type=int, default=100, help="Steps before starting SAC training")
    parser.add_argument("--sac_buffer_size", type=int, default=10000)
    parser.add_argument("--sac_batch_size", type=int, default=32)
    parser.add_argument("--sac_updates_per_interval", type=int, default=2)
    parser.add_argument("--actor_lr", type=float, default=2e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.2, help="SAC temperature parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update rate")
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="log_ratio_asym",
        choices=["log_ratio_asym", "budget_pace_log", "epsilon_guard", "soft_tanh"],
        help="Reward function mode"
    )
    parser.add_argument(
        "--disable_rl",
        action="store_true",
        help="Disable RL, use fixed noise/uniform layer weights as baseline"
    )
    parser.add_argument(
        "--no_dp",
        action="store_true",
        help="Completely disable differential privacy (no noise added, for control experiment)"
    )

    # Privacy Parameters
    parser.add_argument("--epsilon", type=float, default=12.0, help="Target privacy budget (ε)")
    parser.add_argument("--delta", type=float, default=1e-5, help="Privacy parameter (δ)")
    parser.add_argument("--default_norm_c", type=float, default=4.0, help="Default clipping norm constant")
    parser.add_argument("--default_noise_multiplier", type=float, default=None,
                        help="Default noise multiplier (if None, automatically calculated based on epsilon)")

    # Others
    parser.add_argument("--output_dir", type=str, default="./rap_ldp_output")
    parser.add_argument("--eval_batches_for_state", type=int, default=10, help="Number of eval batches used for state construction")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Specify GPU IDs to use, separated by commas, e.g., '0,1,2'. Achieved by setting CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--use_data_parallel", action="store_true",
                        help="Use DataParallel for multi-GPU training")
    parser.add_argument("--disable_gen_bleu", action="store_true",
                        help="Disable generative BLEU (only e2e_nlg/dart)")
    parser.add_argument("--gen_max_new_tokens", type=int, default=64,
                        help="Maximum generation length for generative BLEU")
    parser.add_argument("--gen_min_new_tokens", type=int, default=5,
                        help="Minimum generation length for generative BLEU")
    parser.add_argument("--gen_log_samples", type=int, default=0,
                        help="Number of samples to print when generating BLEU (0 means no printing)")

    # WandB Parameters
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="RAP-LDP", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (if None, automatically generated)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/team name")

    args = parser.parse_args()
    train_rl_dp_forward(args)
