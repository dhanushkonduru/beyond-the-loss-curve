# Execution Videos

Screen-recorded terminal demonstrations: the code, one live execution, and the measured results, in four short clips.

| File | Content |
|---|---|
| `project_and_methods.mp4` | The project layout and the bounded NPO objective in the code |
| `live_proof.mp4` | `verify_deletion.py` running live — the same question asked of the model before and after unlearning |
| `results.mp4` | The measured results: `seeds_summary.txt`, `relearn_report.txt`, `mia_report.txt`, `sync_report.txt` |
| `reproducibility.mp4` | `run_pipeline.sh` and `run_multiseed.sh` — the full study is two scripts |

The script behind the live-execution clip, `verify_deletion.py`, is in [`../2_Source code/pipeline/`](../2_Source%20code/pipeline/). It asks one deleted record and one retained record of the model before and after unlearning, then recomputes the two headline metrics (UES, KRR) over a small sample — so the numbers shown may vary slightly run to run. The stable, three-seed figures are the ones reported in the manuscript.
