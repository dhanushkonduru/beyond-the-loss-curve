# Beyond the Loss Curve

**Behavioural and Adversarial Verification of Machine Unlearning in Language Models**

Dhanush Konduru, Rajasekhar Babu M, Nirmala M

Machine unlearning is judged almost entirely by a rising loss on the deleted records — but a loss curve reports how surprised a model is by one exact string, not whether the underlying fact can still be recovered by asking differently, or whether it is genuinely gone rather than just suppressed. This repository holds the figures, the full source code, the measured results, the plagiarism/AI-writing integrity reports and the execution videos for a framework that closes that gap: it interrogates every deleted record through five differently-phrased probes, detects leakage by searching answers for the record's true values, and combines the evidence into an unlearning effectiveness score, a knowledge retention ratio, a behavioural consistency measure and a single trustworthiness score — then attacks its own verdict with membership inference and a brief relearning attempt, graded against a reference model retrained without the deleted data. The manuscript text itself is kept private ahead of submission and is not included here.

## Folders

| Folder | Contents |
|---|---|
| [`0_Manuscript&PlagiarismReports/`](0_Manuscript%26PlagiarismReports/) | Turnitin similarity and AI-writing reports (0% similarity, 20% AI-detected). The manuscript itself is kept out of this public repository. |
| [`1_Diagrams/`](1_Diagrams/) | The eleven figures, exactly as cited in the paper |
| [`2_Source code/`](2_Source%20code/) | The full framework — pipeline, web console, figure scripts |
| [`3_Execution videos/`](3_Execution%20videos/) | Terminal recordings: code, a live execution, and the results |
| [`4_Datasets/`](4_Datasets/) | The WHO benchmark and the two versioned releases, nothing synthetic |
| [`5_Any other/`](5_Any%20other/) | Every measured result behind the paper, and its markdown source |

Each folder has its own `README.md` with more detail.

## Headline results (three seeds, mean)

| | Gradient Ascent | Gradient Difference | NPO |
|---|---|---|---|
| Trustworthiness (TS) | 0.125 | 0.340 | **0.501** |
| Privacy leakage (PLI) | +0.345 | +0.336 | **−0.016** |
| Recovered by relearning attack | 87.9% | 79.5% | 95.2% |

The retrained reference model — which never saw the deleted data — recovers only **28.1%** under the same relearning attack. That gap is the central finding: every approximate unlearning objective tested here suppresses the deleted knowledge rather than erasing it, a fact invisible to a training loss curve but immediately visible once the model is asked the right question.

## Quick start

```bash
cd "2_Source code/pipeline"
pip install -r requirements.txt
python generate_dataset.py
EPOCHS=15 python train.py
bash run_pipeline.sh              # train -> unlearn (3 methods) -> oracle -> audit -> attacks
python verify_deletion.py         # 13-second behavioural proof of deletion
```

## Data availability

All source data is real: national immunisation coverage and tuberculosis incidence indicators published by the WHO Global Health Observatory (<https://www.who.int/data/gho>), retrieved through the public API at `https://ghoapi.azureedge.net/api`. The derived benchmark is deterministic given the source release and the stated seed, so it is reproducible rather than only transferred as static files — see [`4_Datasets/README.md`](4_Datasets/README.md).

## Academic context

Consistent with TOFU (Maini et al., COLM 2024) and MUSE (Shi et al., ICLR 2025): approximate unlearning trades utility for forgetting rather than erasing cleanly, and loss-based evaluation cannot tell the difference. The full study — dataset construction, training, three unlearning objectives, the reference model, the assessment stage and both attacks — runs on a single consumer machine with hardware acceleration.

---
*Final-year AI/ML engineering capstone.*
