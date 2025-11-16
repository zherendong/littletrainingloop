"""
Wrapper for integrating littletrainingloop models with lm-evaluation-harness.

This module provides a bridge between littletrainingloop's TransformerModel
and the lm-evaluation-harness framework for standardized model evaluation.
"""

import torch
from typing import List, Tuple, Optional
import tiktoken

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

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
        Note: In typical usage within littletrainingloop, you will construct
        this wrapper directly in Python (see `evaluate_checkpoint` helper
        below) rather than via the lm-eval CLI.

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

        # Load tokenizer using the same default used for training
        # This keeps tokenization centralized in language_model_dataloader.default_tokenizer()
        self.tokenizer = language_model_dataloader.default_tokenizer()

        # Load model and metadata from a training checkpoint. This centralizes
        # how we interpret checkpoint metadata.
        checkpoint_result = checkpointing.load_model_from_training_checkpoint(
            checkpoint_path, device=device
        )
        self.model = checkpoint_result["model"]
        checkpoint_metadata = checkpoint_result.get("metadata", {})

        # Get vocab_size from metadata if available, otherwise infer from model weights
        if "vocab_size" in checkpoint_metadata:
            self.vocab_size = checkpoint_metadata["vocab_size"]
        else:
            # Infer from embedding weight shape
            embedding_weight = self.model.state_dict()["embedding.weight"]
            self.vocab_size = embedding_weight.shape[0]

        # Ensure tokenizer and model vocabularies are aligned
        if self.tokenizer.n_vocab > self.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({self.tokenizer.n_vocab}) is larger than the "
                f"model vocab size ({self.vocab_size}). "
                "Ensure the model was trained with a compatible tokenizer."
            )

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
        # This should match the model's context window. The transformer currently
        # uses rotary embeddings with max_seq_len=8192.
        return 8192

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

    def loglikelihood_rolling(self, requests) -> List[float]:
        """Compute perplexity on full sequences.

        Used for language modeling benchmarks where we want to compute
        the perplexity of an entire text.

        This implementation is intentionally simple and does **one forward
        pass per document** (rather than one per token) to avoid O(L^2) work
        and a large number of distinct sequence lengths, which interacts badly
        with ``torch.compile``.

        Args:
            requests: List of Instance objects with property `args` returning
                     (text,) tuples (single string).

        Returns:
            List of log probabilities (one float per request).
        """
        results: List[float] = []

        for request in requests:
            # For rolling, we get a single string
            text = request.args[0]

            # Tokenize the full text
            token_ids = self.tok_encode(text)

            if len(token_ids) == 0:
                results.append(0.0)
                continue

            # If sequence is longer than the model context window, keep the
            # most recent tokens so that [EOT] + tokens fits into max_length.
            if len(token_ids) + 1 > self.max_length:
                token_ids = token_ids[-(self.max_length - 1) :]

            # Use a single EOT token as prefix context and compute logprobs
            # for the entire sequence in one shot.
            context_ids = [self.eot_token_id]

            context_tensor = torch.tensor([context_ids], device=self.device)
            target_tensor = torch.tensor([token_ids], device=self.device)

            token_logprobs = self.model.compute_token_logprobs(
                context_tensor, target_tensor
            )
            total_logprob = float(token_logprobs.sum().item())

            results.append(total_logprob)

        return results

    def generate_until(self, requests) -> List[str]:
        """Generate text continuations until stopping criteria are met.

        Used for generative tasks like open-ended QA.

        Args:
            requests: List of Instance objects with property `args` returning
                     (context, gen_kwargs) tuples where gen_kwargs contains
                     generation parameters like 'until' (stop sequences),
                     'max_gen_toks', etc.

        Returns:
            List of generated text strings (continuations only, not including
            context).
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
                    torch.tensor([[next_token_id]], device=self.device),
                ], dim=1)
            else:
                # Max tokens reached without hitting stop sequence
                generated_text = self.tok_decode(generated_ids)
                results.append(generated_text)

        return results



def evaluate_checkpoint(
    checkpoint_path: str,
    tasks: list[str] | None = None,
    limit: int | None = None,
    device: str = "cuda",
):
    """Convenience helper to run lm-eval at the end of training.

    This is intended for programmatic use inside a training script, e.g.:

        results = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            tasks=["hellaswag", "arc_easy"],
            limit=100,
        )

    Args:
        checkpoint_path: Path to a training checkpoint saved with
            ``checkpointing.save_training_checkpoint``.
        tasks: List of lm-eval task names. If None, callers should pass
            tasks explicitly when calling lm_eval.simple_evaluate.
        limit: Optional sample limit per task for quicker smoke tests.
        device: Device to run evaluation on.

    Returns:
        The dictionary returned by ``lm_eval.simple_evaluate``.
    """
    from lm_eval import simple_evaluate

    wrapper = LittleTrainingLoopLM(checkpoint_path=checkpoint_path, device=device)

    if tasks is None:
        raise ValueError("'tasks' must be provided when using evaluate_checkpoint().")

    print("[lm_eval_wrapper] Starting simple_evaluate()", flush=True)
    results = simple_evaluate(
        model=wrapper,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
        device=wrapper.device,
    )
    print("[lm_eval_wrapper] simple_evaluate() returned", flush=True)
    return results
