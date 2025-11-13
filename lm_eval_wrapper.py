"""
Wrapper for integrating littletrainingloop models with lm-evaluation-harness.

This module provides a bridge between littletrainingloop's TransformerModel
and the lm-evaluation-harness framework for standardized model evaluation.
"""

import torch
from typing import List, Tuple, Optional
import tiktoken

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:
    # If lm_eval is not installed, create dummy base class
    print("Warning: lm-evaluation-harness not installed. Install with: pip install lm-eval")
    class LM:
        pass
    def register_model(name):
        def decorator(cls):
            return cls
        return decorator

import checkpointing
import transformer
import language_model_dataloader


@register_model("littletrainingloop")
class LittleTrainingLoopLM(LM):
    """
    Wrapper class for littletrainingloop models to work with lm-evaluation-harness.
    
    Usage:
        # From command line:
        lm_eval --model littletrainingloop \
                --model_args checkpoint_path=/path/to/checkpoint.pt \
                --tasks hellaswag,arc_easy \
                --device cuda \
                --batch_size 8
        
        # From Python:
        from lm_eval_wrapper import LittleTrainingLoopLM
        model = LittleTrainingLoopLM(checkpoint_path="/path/to/checkpoint.pt")
        # Use with lm_eval.simple_evaluate()
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        """
        Initialize the wrapper with a checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            device: Device to run the model on ("cuda" or "cpu")
            batch_size: Batch size for inference (currently only 1 is supported)
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._batch_size = batch_size

        # Load tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Load model from checkpoint
        # We need to infer the config from the checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract config and vocab_size from metadata if available
        if "metadata" in checkpoint_data and "config" in checkpoint_data["metadata"]:
            config_data = checkpoint_data["metadata"]["config"]
            # Handle both dict and TransformerConfig object
            if isinstance(config_data, dict):
                # If it's a dict from save_training_checkpoint, extract model_config
                if "model_config" in config_data:
                    model_config = config_data["model_config"]
                    # If model_config is also a dict, convert to TransformerConfig
                    if isinstance(model_config, dict):
                        model_config = transformer.TransformerConfig(**model_config)
                else:
                    # Assume the dict itself is the model config
                    model_config = transformer.TransformerConfig(**config_data)
            else:
                # It's already a TransformerConfig object
                model_config = config_data
        else:
            # Try to infer from model state dict
            raise ValueError(
                "Checkpoint must contain model config in metadata. "
                "Please save checkpoints with config information."
            )

        # Get vocab_size from metadata if available, otherwise use tokenizer
        if "metadata" in checkpoint_data and "vocab_size" in checkpoint_data["metadata"]:
            self.vocab_size = checkpoint_data["metadata"]["vocab_size"]
        else:
            # Infer from embedding weight shape
            embedding_weight = checkpoint_data["model_state_dict"]["embedding.weight"]
            self.vocab_size = embedding_weight.shape[0]

        # Create model
        self.model = transformer.TransformerModel(self.vocab_size, model_config)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        # Set properties required by lm_eval
        self._rank = 0
        self._world_size = 1
    
    @property
    def eot_token_id(self) -> int:
        """End of text token ID."""
        # If the tokenizer's EOT token is out of bounds for our vocab,
        # use a fallback (e.g., 0 or vocab_size - 1)
        eot = self.tokenizer.eot_token
        if eot >= self.vocab_size:
            # Use 0 as EOT for small vocab models
            return 0
        return eot
    
    @property
    def max_length(self) -> int:
        """Maximum sequence length the model can handle."""
        # This should match the model's context window
        # For now, return a reasonable default
        return 2048
    
    @property
    def max_gen_toks(self) -> int:
        """Maximum number of tokens to generate."""
        return 256
    
    @property
    def batch_size(self) -> int:
        """Batch size for inference."""
        return self._batch_size
    
    @property
    def device(self) -> str:
        """Device the model is running on."""
        return self._device
    
    @device.setter
    def device(self, value: str):
        """Set the device."""
        self._device = value
    
    def tok_encode(self, string: str) -> List[int]:
        """
        Tokenize a string into token IDs.
        
        Args:
            string: Text to tokenize
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(string, allowed_special="all")
    
    def tok_decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of generating continuations from contexts.

        This is the core method for multiple-choice and completion tasks.

        Args:
            requests: List of Instance objects with property `args` returning
                     (context, continuation) tuples.

        Returns:
            List of (log_prob, is_greedy) tuples where:
            - log_prob: log probability of generating the continuation
            - is_greedy: True if continuation is the greedy (argmax) prediction
        """
        results = []

        for request in requests:
            context, continuation = request.args

            # Tokenize context and continuation
            if context == "":
                # Empty context: use EOT token
                context_ids = [self.eot_token_id]
                continuation_ids = self.tok_encode(continuation)
            else:
                context_ids = self.tok_encode(context)
                continuation_ids = self.tok_encode(continuation)

            # Convert to tensors
            context_tensor = torch.tensor([context_ids], device=self.device)
            continuation_tensor = torch.tensor([continuation_ids], device=self.device)

            # Compute log probabilities for the continuation
            logprobs = self.model.compute_token_logprobs(
                context_tensor, continuation_tensor
            )

            # Sum log probabilities across all continuation tokens
            total_logprob = logprobs.sum().item()

            # Check if this is a greedy generation
            is_greedy_tensor = self.model.is_greedy_generation(
                context_tensor, continuation_tensor
            )
            is_greedy = is_greedy_tensor.all().item()

            results.append((total_logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        """
        Compute perplexity on full sequences.

        Used for language modeling benchmarks where we want to compute
        the perplexity of an entire text.

        Args:
            requests: List of Instance objects with property `args` returning
                     (text,) tuples (single string).

        Returns:
            List of (log_prob,) tuples for each request.
        """
        results = []

        for request in requests:
            # For rolling, we get a single string
            text = request.args[0]

            # Tokenize the full text
            token_ids = self.tok_encode(text)

            if len(token_ids) == 0:
                results.append((0.0,))
                continue

            # For rolling evaluation, we compute log-likelihood of each token
            # given all previous tokens
            total_logprob = 0.0

            # Start with EOT token as context
            for i in range(len(token_ids)):
                if i == 0:
                    # First token: condition on EOT
                    context_ids = [self.eot_token_id]
                else:
                    # Subsequent tokens: condition on all previous tokens
                    context_ids = token_ids[:i]

                target_id = token_ids[i]

                # Convert to tensors
                context_tensor = torch.tensor([context_ids], device=self.device)
                target_tensor = torch.tensor([[target_id]], device=self.device)

                # Compute log probability
                logprob = self.model.compute_token_logprobs(
                    context_tensor, target_tensor
                )

                total_logprob += logprob.item()

            results.append((total_logprob,))

        return results

    def generate_until(self, requests) -> List[str]:
        """
        Generate text continuations until stopping criteria are met.

        Used for generative tasks like open-ended QA.

        Args:
            requests: List of Instance objects with property `args` returning
                     (context, gen_kwargs) tuples where gen_kwargs contains
                     generation parameters like 'until' (stop sequences),
                     'max_gen_toks', etc.

        Returns:
            List of generated text strings (continuations only, not including context).
        """
        results = []

        for request in requests:
            context, gen_kwargs = request.args

            # Extract generation parameters
            until = gen_kwargs.get("until", [self.tok_decode([self.eot_token_id])])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            # Tokenize context
            if context == "":
                context_ids = [self.eot_token_id]
            else:
                context_ids = self.tok_encode(context)

            # Convert to tensor
            input_ids = torch.tensor([context_ids], device=self.device)

            # Generate tokens autoregressively
            generated_ids = []

            for _ in range(max_gen_toks):
                # Get logits for next token
                logits = self.model.get_logits(input_ids)

                # Take the last token's logits and get argmax (greedy decoding)
                next_token_logits = logits[0, -1, :]
                next_token_id = next_token_logits.argmax().item()

                # Add to generated sequence
                generated_ids.append(next_token_id)

                # Decode to check for stop sequences
                generated_text = self.tok_decode(generated_ids)

                # Check if we've hit a stop sequence
                should_stop = False
                for stop_seq in until:
                    if stop_seq in generated_text:
                        # Trim the stop sequence from the output
                        generated_text = generated_text.split(stop_seq)[0]
                        should_stop = True
                        break

                if should_stop:
                    results.append(generated_text)
                    break

                # Append next token to input for next iteration
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token_id]], device=self.device)
                ], dim=1)
            else:
                # Max tokens reached without hitting stop sequence
                generated_text = self.tok_decode(generated_ids)
                results.append(generated_text)

        return results

