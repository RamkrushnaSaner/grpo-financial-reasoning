"""Tasks: questions, prompts, answer parsing and verification.

Built-in tasks register themselves on import. Add a task by subclassing
``Task`` and decorating it with ``@TASKS.register("name")``.
"""
from .base import TASKS, Sample, Task, build_task
from .financial import (
    TOLERANCE,
    FinancialTask,
    build_messages,
    extract_answer,
    generate,
    has_answer_tag,
    verify_answer,
)

__all__ = ["TASKS", "Sample", "Task", "build_task", "TOLERANCE", "FinancialTask",
           "build_messages", "extract_answer", "generate", "has_answer_tag", "verify_answer"]
