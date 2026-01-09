"""
Wrapper for integrating littletrainingloop models with lm-evaluation-harness.

This module provides a bridge between littletrainingloop's TransformerModel
and the lm-evaluation-harness framework for standardized model evaluation.
"""

from pathlib import Path
import torch
from typing import List, Tuple

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

import checkpointing
import language_model_dataloader
import lm_eval
import language_model_basics
import dataclasses
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


@dataclasses.dataclass
class Task:
    name: str
    key: str
    task_key_name: str


available_tasks = {
    "hellaswag": Task("hellaswag", "acc_norm,none", "accuracy"),
    "arc_easy": Task("arc_easy", "acc_norm,none", "accuracy"),
    "arc_challenge": Task("arc_challenge", "acc_norm,none", "accuracy"),
    "humaneval": Task("humaneval", "pass@1,create_test", "pass@1"),
    # math benchmarks
    "gsm8k": Task("gsm8k", "exact_match,flexible-extract", "accuracy"),
    "hendrycks_math": Task("hendrycks_math", "exact_match,none", "accuracy"),
    "mmlu": Task("mmlu", "acc,none", "accuracy"),
    "mbpp": Task("mbpp", "pass_at_1,none", "accuracy"),
    "mathqa": Task("mathqa", "acc_norm,none", "accuracy"),
    # other standard tasks
    "piqa": Task("piqa", "acc_norm,none", "accuracy"),
    "sciq": Task("sciq", "acc_norm,none", "accuracy"),
    "winogrande": Task("winogrande", "acc,none", "accuracy"),
    "triviaqa": Task("triviaqa", "exact_match,remove_whitespace", "accuracy"),
    "openbookqa": Task("openbookqa", "acc_norm,none", "accuracy"),
    "drop": Task("drop", "f1,none", "accuracy"),
    "bbh": Task("bbh", "acc_norm,none", "accuracy"),
    "naturalquestions": Task("nq_open", "exact_match,remove_whitespace", "accuracy"),
    "agieval": Task("agieval", "acc_stderr,none", "accuracy"),
    "race": Task("race", "acc_norm,none", "accuracy"),
    # perplexity-based measurements
    "pile": Task("pile", "perplexity,none", "perplexity"),
    # chinese
    # "cmmlu": Task("cmmlu", "acc_norm,none", "accuracy"),
    "ceval-valid": Task("ceval-valid", "acc_norm,none", "accuracy"),
    # multi lingual
    "kmmlu": Task("kmmlu", "acc,none", "accuracy"),
    # "french_bench": Task("french_bench", "acc_norm,none", "accuracy"),
    # "japanese_leaderboard": Task("japanese_leaderboard", "acc_norm,none", "accuracy"),
    # unused
    "lambada": Task("lambada", "acc_norm,none", "accuracy"),
    "pubmedqa": Task("pubmedqa", "acc_norm,none", "accuracy"),
    "strategyqa": Task("strategyqa", "acc_norm,none", "accuracy"),
    "qasc": Task("qasc", "acc_norm,none", "accuracy"),
    "socialiqa": Task("socialiqa", "acc_norm,none", "accuracy"),
    "commonsenseqa": Task("commonsenseqa", "acc_norm,none", "accuracy"),
    "xquad": Task("xquad", "f1,none", "accuracy"),
    "spanish_bench": Task("spanish_bench", "acc_norm,none", "accuracy"),
}

default_tasks = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "humaneval",
    "gsm8k",
    "hendrycks_math",
    "mmlu",
    "piqa",
    "sciq",
    "winogrande",
    "triviaqa",
    "openbookqa",
    "drop",
    "ceval-valid",
    "bbh",
    "naturalquestions",
    "mbpp",
    "agieval",
    "race",
    "pile",
    "xquad",
    "kmmlu",
    "spanish_bench",
]


def get_task_details(task_name: str) -> Task:
    """Get the task details for a given task name."""
    return available_tasks[task_name]


@register_model("littletrainingloop")
class LittleTrainingLoopWrapper(LM):
    """
    Wrapper class for littletrainingloop models to work with lm-evaluation-harness.

    Usage:
        ```
        from lm_eval_wrapper import LittleTrainingLoopLM
        from pathlib import Path
        model = LittleTrainingLoopLM(checkpoint_path=Path("/path/to/checkpoint.pt"))
        # Use with lm_eval.simple_evaluate()
        ```
    """

    def __init__(
        self,
        model: language_model_basics.LanguageModel,
        config: language_model_basics.LanguageModelTrainingConfig,
        device: str = "cuda",
        batch_size: int = 1,
        generate_until_max_length: int | None = None,
    ):
        """
        Initialize the wrapper with a checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            device: Device to run the model on ("cuda" or "cpu")
            batch_size: Batch size for inference (currently only 1 is supported)
            generate_until_max_length: Maximum length to generate until
                (default: 256)
        """
        super().__init__()

        self.model = model
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.generate_until_max_length = generate_until_max_length or 256
        self.vocab_size = config.vocab_size

        # Load tokenizer using the same default used for training
        # This keeps tokenization centralized in language_model_dataloader.default_tokenizer()
        self.tokenizer = language_model_dataloader.default_tokenizer()

        def is_illegal_token(token: int) -> bool:
            try:
                self.tokenizer.decode([token])
                return False
            except KeyError:
                return True

        self.illegal_tokens = set(
            filter(is_illegal_token, range(self.tokenizer.n_vocab))
        )

        # Ensure tokenizer and model vocabularies are aligned
        if self.tokenizer.n_vocab > self.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({self.tokenizer.n_vocab}) is larger than the "
                f"model vocab size ({self.vocab_size}). "
                "Ensure the model was trained with a compatible tokenizer."
            )

    @property
    def eot_token_id(self) -> int:
        """End of text token ID."""
        eot = self.tokenizer.eot_token
        if eot >= self.vocab_size:
            raise ValueError(
                f"Tokenizer EOT token ({eot}) is out of bounds for the model vocab "
                f"size ({self.vocab_size}). "
                "Ensure the model was trained with a compatible tokenizer."
            )
        return eot

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
        if self.eot_token_id in tokens:
            tokens = tokens[: tokens.index(self.eot_token_id)]

        # Remove illegal tokens
        tokens = [t for t in tokens if t not in self.illegal_tokens]
        try:
            decoded = self.tokenizer.decode(tokens)
        except KeyError as e:
            print(f"Could not decode tokens: {e}")
            decoded = ""
        return decoded

    @torch.inference_mode()
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
            else:
                context_ids = self.tok_encode(context)
            continuation_ids = self.tok_encode(continuation)

            # Convert to tensors
            context_tokens = torch.tensor([context_ids], device=self.device)
            continuation_tokens = torch.tensor([continuation_ids], device=self.device)

            # Compute log probabilities for the continuation
            logprobs, is_greedy_per_token = self._get_logprobs(
                context_tokens, continuation_tokens
            )

            # Sum log probabilities across all continuation tokens
            total_logprob = logprobs.sum().item()

            is_greedy = is_greedy_per_token.all().item()

            results.append((total_logprob, is_greedy))

        return results

    @torch.inference_mode()
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
            if len(token_ids) + 1 > self.config.sequence_length:
                token_ids = token_ids[-(self.config.sequence_length - 1) :]

            # Use a single EOT token as prefix context and compute logprobs
            # for the entire sequence in one shot.
            context_ids = [self.eot_token_id]

            context_tokens = torch.tensor([context_ids], device=self.device)
            target_tokens = torch.tensor([token_ids], device=self.device)

            token_logprobs, _ = self._get_logprobs(context_tokens, target_tokens)
            total_logprob = float(token_logprobs.sum().item())

            results.append(total_logprob)

        return results

    @torch.inference_mode()
    def _get_logprobs(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for target tokens.

        Note: Currently designed for batch_size=1 (no padding/masking support yet).

        Args:
            input_ids: Context token IDs of shape [batch_size, context_len]
                where context_len >= 1
            target_ids: Target token IDs of shape [batch_size, target_len]

        Returns:
            Log probabilities for each target token, shape [batch_size, target_len]
        """
        batch_size, context_len = input_ids.shape
        target_len = target_ids.shape[1]
        seq_len = context_len + target_len
        assert (
            batch_size == 1
        ), "Batch size > 1 not supported yet. At least that's what the documentation claims."
        if context_len < 1:
            raise ValueError(
                f"input_ids must have at least 1 context token, got {context_len}. "
                "Consider prepending a BOS token if context is empty."
            )

        # Concatenate context and targets
        full_input = torch.cat([input_ids, target_ids], dim=1)

        # Get logits
        logits = self.model.forward(
            full_input,
            use_optimized=False,  # otherwise we trigger cuda graph recompilation
        )  # [batch, seq_len, vocab]

        # Extract logits for positions where we predict target tokens.
        # We exclude the final token (at position context_len + target_len - 1)
        # because it is EOS.
        assert logits.shape[1] == seq_len
        prediction_logits = logits[:, context_len - 1 : -1, :]
        assert prediction_logits.shape == (batch_size, target_len, self.vocab_size), (
            f"Expected prediction_logits to have shape "
            f"({batch_size}, {target_len}, {self.vocab_size}), got {prediction_logits.shape}"
        )

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(prediction_logits, dim=-1)

        # Gather log probs for actual target tokens
        batch_indices = torch.arange(batch_size, device=target_ids.device).unsqueeze(1)
        position_indices = torch.arange(target_len, device=target_ids.device).unsqueeze(
            0
        )
        token_logprobs = log_probs[batch_indices, position_indices, target_ids]

        # is_greedy = self.model.is_greedy_generation(input_ids, target_ids)
        is_greedy = torch.zeros_like(token_logprobs, dtype=torch.bool)
        greedy_tokens = prediction_logits.argmax(dim=-1)
        is_greedy = greedy_tokens == target_ids

        return token_logprobs, is_greedy

    def infer(self, context: str, until: list[str]):
        # Extract generation parameters
        # until = gen_kwargs.get("until", [self.tok_decode([self.eot_token_id])])

        assert context != "", "Context must not be empty"
        context_ids = self.tok_encode(context)

        # Convert to tensor
        input_ids = torch.tensor([context_ids], device=self.device)
        assert input_ids.shape == (1, len(context_ids))

        # Generate tokens autoregressively
        generated_ids = []

        for _ in range(self.generate_until_max_length):
            # asser both model and input_ids are on GPU
            assert input_ids.device.type == "cuda"
            assert next(self.model.parameters()).device.type == "cuda"

            # Get logits for next token
            logits = self.model.forward(input_ids, use_optimized=False)

            # Take the last token's logits and get argmax (greedy decoding)
            next_token_logits = logits[0, -1, :]
            next_token_id = next_token_logits.argmax().item()

            # Add to generated sequence
            generated_ids.append(next_token_id)

            # Decode to check for stop sequences
            generated_text = self.tok_decode(generated_ids)

            # check for EOS token
            if next_token_id == self.eot_token_id:
                return generated_text

            # Check if we've hit a stop sequence
            for stop_seq in until:
                if stop_seq in generated_text:
                    # Trim the stop sequence from the output
                    generated_text = generated_text.split(stop_seq)[0]
                    return generated_text

            # Append next token to input for next iteration
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[next_token_id]], device=self.device),
                ],
                dim=1,
            )
        else:
            # Max tokens reached without hitting stop sequence
            generated_text = self.tok_decode(generated_ids)
            return generated_text

    @torch.inference_mode()
    def generate_until(self, requests) -> List[str]:
        """Generate text continuations until stopping criteria are met.

        Used for generative tasks like open-ended QA.

        Args:
            requests: List of Instance objects with property `args` returning
                     (context, gen_kwargs) tuples where gen_kwargs contains
                     generation parameters like 'until' (stop sequences),
                     'self.generate_until_max_length', etc.

        Returns:
            List of generated text strings (continuations only, not including
            context).
        """
        results = []

        for request in requests:
            context, gen_kwargs = request.args
            until = gen_kwargs.get("until", [self.tok_decode([self.eot_token_id])])
            generated_text = self.infer(context, until)
            results.append(generated_text)

        return results


def evaluate_model(
    model: language_model_basics.LanguageModel,
    config: language_model_basics.LanguageModelTrainingConfig,
    tasks: list[str],
    limit: int | None = None,
    device: str = "cuda",
    generate_until_max_length: int | None = None,
):
    """Convenience helper to run lm-eval at the end of training.

    This is intended for programmatic use inside a training script, e.g.:

        results = evaluate_model(
            model=model,
            config=config,
            tasks=["hellaswag", "arc_easy"],
        )

    Args:
        model: The model to evaluate.
        config: The training config.
        tasks: List of lm-eval task names.
        limit: Optional sample limit per task for quicker smoke tests.
        device: Device to run evaluation on.
        generate_until_max_length: Maximum length to generate until

    Returns:
        The dictionary returned by ``lm_eval.simple_evaluate``.
    """
    wrapper = LittleTrainingLoopWrapper(
        model=model,
        config=config,
        device=device,
        batch_size=config.eval_config.batch_size,
        generate_until_max_length=generate_until_max_length,
    )

    print("[lm_eval_wrapper] Starting simple_evaluate()", flush=True)
    # documentation:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py
    results = lm_eval.simple_evaluate(  # type: ignore
        model=wrapper,
        tasks=tasks,
        # num_fewshot=0,  # what is this?
        limit=limit,
        device=wrapper.device,
        max_batch_size=wrapper.batch_size,
        cache_requests=True,
        confirm_run_unsafe_code=True,
    )
    print("[lm_eval_wrapper] simple_evaluate() returned", flush=True)
    return results


def evaluate_checkpoint(
    checkpoint_path: Path,
    tasks: list[str] | None = None,
    limit: int | None = None,
    device: str = "cuda",
    generate_until_max_length: int | None = None,
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
        tasks: List of lm-eval task names.
        limit: Optional sample limit per task for quicker smoke tests.
        device: Device to run evaluation on.
        generate_until_max_length: Maximum length to generate until

    Returns:
        The dictionary returned by ``lm_eval.simple_evaluate``.
    """
    checkpoint = checkpointing.load_model_from_training_checkpoint(
        checkpoint_path, device=device
    )

    if tasks is None:
        tasks = default_tasks

    results = evaluate_model(
        model=checkpoint.model,
        config=checkpoint.config,
        tasks=tasks,
        limit=limit,
        device=device,
        generate_until_max_length=generate_until_max_length,
    )
    return results
