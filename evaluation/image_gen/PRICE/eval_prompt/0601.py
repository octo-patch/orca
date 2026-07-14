MIN_SCORE = 1
MAX_SCORE = 5

EVAL_PROMPT_TEMPLATE = """You are a practical benchmark evaluator using lenient pass criteria. You will receive two images (an original image and a modified image) along with a specific modification instruction.

Modification instruction:
{instruction}

The first image is the original image. The second image is the modified image.

Score Instruction Following from 1 to 5 (integers only). Score higher when the modified image clearly moves in the direction requested by the instruction. The instruction does not need to be fully completed: visible progress, an in-progress state, or a plausible trend toward the requested outcome should receive credit.

General scoring philosophy:
- Use the full 1-5 range to reflect how well the instruction is followed.
- Object deformation is acceptable: an existing object may change shape, pose, texture, or appearance between the original and modified image.
- Do NOT accept hallucinated additions: strongly penalize cases where the modified image contains a new object that was not present in the original image and was not required by the instruction.
- Minor blur, texture shifts, small artifacts, or partial ambiguity should NOT automatically fail.

Respond with JSON only, no markdown fences:
{{
  "score": 5,
  "reasoning": "..."
}}"""