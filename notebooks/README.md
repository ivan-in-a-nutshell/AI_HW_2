# Notebooks

Run in order. Code lives in `src/visdet/`, experiment settings in `configs/`, and every
evaluated run is logged to `experiments/results.csv` (per-run metrics, history and config in
`experiments/runs/<run>/`).

| # | notebook | where | what | GPU-h |
|---|---|---|---|---|
| 01 | `01_eda.ipynb` | local, CPU | data understanding → design decisions; report figures | 0 |
| 02 | `02_prepare_data.ipynb` | Kaggle, CPU | test-scale frames, masked ignore regions, YOLO labels, folds | 0 |
| 03 | `03_two_stage.ipynb` | Kaggle, 2× T4 | Faster R-CNN: baseline + 3 tweaks, RPN vs RoI-head analysis, failure cases | ~8 |
| 04 | `04_one_stage_anchor.ipynb` | Kaggle, 2× T4 | RetinaNet: baseline + 3 tweaks, comparison with Faster R-CNN | ~8 |
| 05 | `05_anchor_free.ipynb` | Kaggle, 2× T4 | YOLO11s: resolution, P2 head, TAL top-k; first submission | ~10 |
| 06 | `06_query_based.ipynb` | Kaggle, 2× T4 | RT-DETR: queries, matching cost, backbone LR; curves vs YOLO | ~8 |
| 07 | `07_pipeline.ipynb` | Kaggle, 2× T4 | data/inference vs model experiments, OOF check, final submissions | ~25 |

## Kaggle set-up

1. **Code.** Each notebook's first cell clones `GITHUB_REPO` (set it once; for a private repository
   add a `GITHUB_TOKEN` secret under *Add-ons → Secrets*). Alternatively, upload the repository
   folder as a Kaggle dataset and attach it: the cell picks up any input containing `pyproject.toml`.
2. **Inputs.** The competition data in every notebook. The output of notebook 02 in notebooks 03–07.
   The outputs of notebooks 05 and 06 in notebook 07, which needs their checkpoints.
3. **Session.** Accelerator *GPU T4 ×2*, Internet on, *Save & Run All*. Runs that already have a
   checkpoint in `/kaggle/working/runs` are skipped, so an interrupted notebook can be resumed.
4. **Afterwards.** Download `repo/experiments/` and `repo/images/` from the notebook output and commit
   them. Later notebooks and the report read the shared log from there.

## Local CPU dry run

Every notebook runs end to end on a laptop in a few minutes with 1 epoch on a handful of frames.
Use this to check code before spending GPU time. Logs and figures go to a scratch folder:

```bash
pip install -e . -r requirements.txt torch torchvision
python -m visdet.prepare --out prepared --step 10        # local subset (every 10th frame)
VISDET_SMOKE=1 VISDET_EXPERIMENTS=/tmp/visdet VISDET_FIGURES=/tmp/visdet \
  jupyter nbconvert --to notebook --execute notebooks/03_two_stage.ipynb --output-dir /tmp/visdet
```
