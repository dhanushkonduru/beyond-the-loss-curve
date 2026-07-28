# Diagrams

The eleven figures exactly as the manuscript numbers and cites them, each rendered at 2000 × 1500 px (3.0 MP). Kept as separate files because most journal submission systems, including IEEE Access, want figures uploaded individually rather than only embedded in the document.

| File | Figure | Section |
|---|---|---|
| `fig1_architecture.png` | Fig. 1 — system architecture, three layers | III-B |
| `fig2_workflow.png` | Fig. 2 — unlearning and verification workflow | III-D |
| `fig3_prompt_pipeline.png` | Fig. 3 — multi-prompt behavioural assessment pipeline | IV-B |
| `fig4_metric_framework.png` | Fig. 4 — trustworthiness evaluation framework | V |
| `fig5_benchmark_design.png` | Fig. 5 — experimental benchmark design | VI-A |
| `fig6_trust.png` | Fig. 6 — UES / KRR / TS across three seeds | VII-B |
| `fig7_tradeoff.png` | Fig. 7 — forgetting versus utility trade-off | VII-C |
| `fig8_threat_model.png` | Fig. 8 — threat model | VIII-A |
| `fig9_pli.png` | Fig. 9 — residual privacy leakage | VIII-B |
| `fig10_relearn.png` | Fig. 10 — knowledge recovered by relearning | VIII-C |
| `fig11_roadmap.png` | Fig. 11 — research roadmap | XI |

The scripts that generate these are in [`../2_Source code/paper_figures/`](../2_Source%20code/paper_figures/) — `paper_figs.py` for the seven conceptual diagrams (1–5, 8, 11) and `aggregate_seeds.py` for the four data-driven figures (6, 7, 9, 10).
