# AI Review Guidelines for Experiments

When reviewing or assisting with experiments in this directory, AI assistants should:

## 0. Before Any Action (Checklist)

Before running code, installing packages, or starting long operations, check:
- Is this >10K items or >20 lines of code? → Write to a script file
- Could this take >10 seconds? → Estimate time first, ask before proceeding
- Am I installing/modifying packages? → Ask permission first
- Am I about to commit/push/deploy? → Ask permission first
- Can I simplify the experiment? → Suggest ways to make it simpler/faster

## 1. Consistency

- If we revise the experiment plan, make sure to update the spec.md file to
  reflect the changes. This may bring up new questions; don't hesitate to ask.
- If you see something, say something. We don't want potential risk factors being overlooked.

## 2. Highlight Statistical Caveats

Before drawing conclusions from any experiment, check for:

- **Sample size**: Is the sample large enough to be representative?
- **Sample diversity**: Are samples drawn from diverse sources, or concentrated in a few files/texts?
- **Selection bias**: Are we selecting for certain characteristics (e.g., long texts only)?
- **Seed sensitivity**: Would results change significantly with a different random seed?

**Example of a problematic setup:**
> "Using 5 texts from a single file" - This introduces correlation between samples
> and may not represent the full data distribution.

**Better approach:**
> "Using 500 windows, each from a random position in a random file" - This ensures
> independence and diversity across the sample.

## 3. Document Assumptions

Every experiment should explicitly state:

- What assumptions are being made
- What would invalidate those assumptions
- What the expected outcome is before running (to avoid confirmation bias)

## 4. Reproducibility

Ensure experiments:

- Write their code to a file instead of just running it as a one-off, 
  especially if it's more than a couple of lines.
- Use fixed random seeds (feel free to use prng.py for this)
- Document all parameters
- Can be re-run with different seeds to check robustness
- Save results to files rather than just printing to terminal

## 5. Sanity Checks

Before trusting results:

- Do the results match intuition? If not, why?
- Are there edge cases that weren't tested?
- What's the variance/uncertainty in the measurements?

## 6. Scope Creep

When an experiment grows in complexity:

- Consider splitting into multiple focused experiments
- Document what each sub-experiment is testing
- Avoid mixing validation of different hypotheses

## 7. Code Hygiene

- Do not keep around dead code.

## 8. Dependencies

- **Never install or modify packages without explicit user permission.**
- Avoid using HuggingFace libraries (transformers, tokenizers, etc.) where possible.
- When HuggingFace is unavoidable (e.g., sentence-transformers), contain it to a
  single module and don't let it spread throughout the codebase.

## 9. Extending claude.md files

- claude.md files are meant to contain best practices and lessons learned from
  previous experiments. 
- If you notice that the same problems keeps coming up in multiple experiments,
  suggest ways to extend the claude.md files to cover these patterns. Think hard
  to generalize and make the instructions reusable yet actionable.
- When coming up with changes to the claude.md files, consider the following
  questions:
  - What is the problem that we're trying to solve?
  - Why is it a problem?
  - How did it manifest?
  - What is the root cause?
  - What is the best way to prevent it in the future?
  - How can we detect it early in the future?
  - How will changes in claude.md affect other experiments?
  - Are there any risks or downsides to the proposed changes?
  - Are existing instructions responsible for the problem? If so, how should they be
    revised?
- Avoid including overly specific examples in claude.md files.