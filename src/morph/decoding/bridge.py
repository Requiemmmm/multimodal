import torch


def geometric_bridge(teacher_logprobs: torch.Tensor, student_logprobs: torch.Tensor, alpha: float) -> torch.Tensor:
    """Local product-of-experts bridge over already processed log probabilities."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if teacher_logprobs.shape != student_logprobs.shape:
        raise ValueError("teacher and student distributions must have equal shape")
    teacher_finite = torch.isfinite(teacher_logprobs)
    student_finite = torch.isfinite(student_logprobs)
    if not torch.equal(teacher_finite, student_finite):
        raise ValueError("teacher/student hard support differs; choose a same-support context or oracle")
    # Multiplication by zero and -inf produces NaN, so preserve exact
    # endpoints rather than relying on the arithmetic form.
    if alpha == 1.0:
        return teacher_logprobs.clone()
    if alpha == 0.0:
        return student_logprobs.clone()
    out = alpha * teacher_logprobs + (1.0 - alpha) * student_logprobs
    return out.masked_fill(~teacher_finite, float("-inf"))
