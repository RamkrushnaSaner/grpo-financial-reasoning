"""GRPO from first principles: a small, verified implementation of Group
Relative Policy Optimization, organised so that tasks, rewards, advantage
estimators, losses and model loaders can be swapped from a config file.

    grpo.config      typed YAML configuration
    grpo.registry    name -> implementation lookup (the extension mechanism)
    grpo.data        tasks: questions, prompts, parsing, verification
    grpo.rewards     reward components and their weighted combination
    grpo.algorithm   advantages, losses, rollouts, the GRPO update step
    grpo.model       the LoRA policy and model loaders
    grpo.training    the training loop
    grpo.evaluation  held-out evaluation, probes, paired comparison, reports
    grpo.experiment  run directories, caching, the end-to-end pipeline
    grpo.tracking    logging and metrics streams
    grpo.stats       pass@k and paired statistics
    grpo.validation  mathematical correctness checks

Importing the package does not import transformers, peft or bitsandbytes; only
the ``hf_lora`` model loader does, when a model is loaded.
"""
__version__ = "2.0.0"

from .config import Config, load_config
from .registry import Registry

__all__ = ["Config", "Registry", "__version__", "load_config"]
