from typing import Literal

Position = Literal["prefix", "infix", "user_suffix"]


def insert_trigger(prompt: str, trigger: str, position: Position = "user_suffix", *, infix_at: int | None = None) -> str:
    """Insert ordinary text at an explicit prompt position.

    ``user_suffix`` is the MORPH default and is deliberately not allowed to
    write model control tokens; callers own any policy-level validation.
    """
    if not isinstance(prompt, str) or not isinstance(trigger, str):
        raise TypeError("prompt and trigger must be strings")
    if position == "prefix":
        return trigger + prompt
    if position == "user_suffix":
        return prompt + trigger
    if position == "infix":
        index = len(prompt) if infix_at is None else infix_at
        if not 0 <= index <= len(prompt):
            raise ValueError("infix_at must be within prompt boundaries")
        return prompt[:index] + trigger + prompt[index:]
    raise ValueError(f"unknown trigger position: {position!r}")
