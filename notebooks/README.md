# Notebooks

| notebook | purpose |
|---|---|
| `quickstart_colab.ipynb` | Run the package on Colab: install, tests, smoke runs, a full run, the report, one-question comparison, ablations |
| `archive/grpo_master.ipynb` | The self-contained teaching notebook that produced Run 6, preserved **exactly as run**. Every section explains intuition, mathematics and implementation before the code |

`archive/grpo_master.ipynb` is a historical artifact. Its statements superseded by later findings are listed in [`docs/LESSONS.md`](../docs/LESSONS.md): it attributes the baseline drop to 112-token truncation (runs 4 and 6 contradict this), it describes clipping as exactly inert at μ=1 (true only with `lora_dropout: 0.0`; it used 0.05), its significance verdicts use the standard error of a single proportion, and its dataset has the growth-label bug fixed in dataset version 2. The `src/grpo` package is the maintained implementation; `configs/run6.yaml` reproduces the notebook's settings.
