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
    "bbh": Task("bbh", "exact_match,get-answer", "accuracy"),
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
    # Synthetic spelling benchmark (count, index, reverse tasks)
    "spelling_bee": Task("spelling_benchmark/spelling_bee.yaml", "exact_match,none", "accuracy"),
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
        batch_size: int = 256,
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

    @staticmethod
    def batch_iterator(*local_iterables, batch_size):
        for i in range(0, len(local_iterables[0]), batch_size):
            yield tuple(lst[i:i+batch_size] for lst in local_iterables)

    def pad_batch(self, tokens: List[torch.Tensor], pad_to_pow2: bool=True) -> torch.Tensor:
        def next_power_of_2(x):
            return 1 if x == 0 else 2**(x - 1).bit_length()

        max_len = 0
        for seq in tokens:
            max_len = max(max_len, seq.shape[0])

        if pad_to_pow2:
            max_len = next_power_of_2(max_len)

        # create new batch tensor of desired shape and same dtype / device as tokens[0]
        batch_size = len(tokens)
        trailing_dims = tokens[0].shape[1:]
        padded_seqs = tokens[0].new_full((batch_size, max_len, *trailing_dims), self.eot_token_id)

        # fill actual token ids from the left
        for i, seq in enumerate(tokens):
            length = seq.shape[0]
            padded_seqs[i, :length, ...] = seq

        return padded_seqs

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
        tokenized_requests, context_lengths, full_text_lengths = [], [], []
        for request in requests:
            context, continuation = request.args

            # Tokenize context and continuation
            if context == "":
                # Empty context: use EOT token
                context_ids = [self.eot_token_id]
            else:
                context_ids = self.tok_encode(context)
            continuation_ids = self.tok_encode(continuation)

            # record which parts of sequence are context and continuation
            context_lengths.append(len(context_ids))
            full_text_lengths.append(len(context_ids)+len(continuation_ids))
            full_seq = context_ids + continuation_ids

            # if enforcing a maximum sequence length, split samples here before adding to tokenized_requests
            # and add logic below to combine for total logprobs / is greedy results.

            # Convert to tensors
            tokenized_requests.append(torch.tensor(full_seq))

        all_logprobs, all_is_greedy = [], []
        batches = self.batch_iterator(tokenized_requests, context_lengths, full_text_lengths, batch_size=self.batch_size)
        for requests_batch, context_lengths_batch, full_lengths_batch in batches:

            # pad all sequences to maximum length
            padded_requests = self.pad_batch(requests_batch, pad_to_pow2=True)

            # Compute log probabilities for all tokens
            # logprobs[i, j] is logprobs for req[i], token[j]
            # so logprobs[i, 0] is logprob for BOS token (defined as 0)
            logprobs, is_greedy_per_token = self._get_logprobs(padded_requests)
            assert logprobs.shape == padded_requests.shape
            _, seq_len = logprobs.shape

            # Mask to identify continuation tokens
            start_target = torch.tensor(context_lengths_batch).unsqueeze(1) # [batch_size, 1]
            end_target = torch.tensor(full_lengths_batch).unsqueeze(1) # [batch_size, 1]
            seq_indices = torch.arange(seq_len).unsqueeze(0) # [1, seq_len]
            keep_mask = (seq_indices >= start_target) & (seq_indices < end_target) # [batch_size, seq_len]

            # Sum logprobs across continuation tokens only
            target_logprobs = logprobs * keep_mask
            all_logprobs.extend(target_logprobs.sum(dim=-1).tolist())

            # Are the expected continuation tokens generated by greedy sampling?
            target_is_greedy = is_greedy_per_token | ~keep_mask # set unwanted values to 1
            all_is_greedy.extend(target_is_greedy.all(dim=-1).tolist())

        return list(zip(all_logprobs, all_is_greedy))

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
        # This is broken
        raise NotImplementedError("loglikelihood_rolling is broken. Please try again later.")

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
        self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for target tokens.

        Note:   Calling code is responsible for creating the batches
                and for ignoring results associated with padding positions.

        Args:
            tokens: Sequence token IDs of shape [batch_size, max_seq_len]
                where max_seq_len >= 1

        Returns:
            Log probabilities for each target token, shape [batch_size, max_seq_len]
        """
        batch_size, seq_len = tokens.shape

        # Get logits
        logits = self.model.forward(
            tokens,
            use_optimized=False,  # otherwise we trigger cuda graph recompilation
        ) # [batch, seq_len, vocab]

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # [batch, seq_len, vocab]

        # Extract logprobs for expected tokens - careful to align predictions and targets
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1)
        position_indices = torch.arange(seq_len-1, device=tokens.device).unsqueeze(0) # ignore final prediction
        token_logprobs = torch.zeros_like(tokens, device=tokens.device, dtype=log_probs.dtype)
        token_logprobs[:, 1:] = log_probs[:, :-1, :].gather(
            dim=-1, index=tokens[:, 1:].unsqueeze(-1) # ignore first token (not predicted)
        ).squeeze(-1)  # [batch, seq_len]

        # Are expected tokens generated when greedy sampling?
        greedy_tokens = logits.argmax(dim=-1) # [batch, seq_len]
        is_greedy = torch.ones_like(tokens, device=tokens.device)
        is_greedy[:, 1:] = greedy_tokens[:, :-1] == tokens[:, 1:]

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
        had_to_trim = False

        for _ in range(self.generate_until_max_length):
            # asser both model and input_ids are on GPU
            assert input_ids.device.type == "cuda"
            assert next(self.model.parameters()).device.type == "cuda"

            if input_ids.shape[1] >= self.config.sequence_length:
                input_ids = input_ids[:, -self.config.sequence_length + 1 :]
                assert input_ids.shape[1] == self.config.sequence_length - 1
                if not had_to_trim:
                    print(
                        f"Warning: Had to trim input to fit into max sequence length. "
                        f"Input length: {input_ids.shape[1]}; max length: {self.config.sequence_length}"
                    )
                had_to_trim = True

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
