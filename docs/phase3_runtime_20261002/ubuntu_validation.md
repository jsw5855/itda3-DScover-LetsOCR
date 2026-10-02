# Ubuntu 22.04 WSL validation — 2026-10-02

**Local Ubuntu 22.04-compatible validation under WSL 2 on the development PC.
This is NOT organizer hardware timing.** The organizer machine may be faster or
slower; the 2500 s limit must be judged with that margin in mind.

Code validated: the production files of branch
`candidate/final-optimization-20261001` (`ocr_pipeline.py`, `date_parser/*.py`,
`submission_runtime.py`, `predict.ipynb`, `requirements.txt`, weight download
scripts). They are byte-identical, ignoring line endings, to this commit.
Driver: `scripts/ubuntu_validate.sh`.

| Check | Result |
|---|---|
| Environment | Ubuntu 22.04 under WSL 2 |
| Execution | organizer-style `jupyter nbconvert --execute predict.ipynb` |
| CPU affinity | restricted to and verified as `[0,1,2,3]` |
| Network | isolated during inference |
| Output | 500 images → schema-correct 500-row CSV (`image_id,year,month,day,final_date`) |
| Wall time | 442.7 s |
| Accuracy sanity check | 438/500 = 87.6% (food development images; consistency check only) |
| Peak RSS | 3.53 GB |
| Peak PSS | 3.22 GB |
| FTZ/DAZ | verified active |
| FTZ on vs off | 0/40 prediction differences |
| Notebook | no errors |

No production change was recommended by this validation. Run outputs (CSV, executed notebook, venv, weights) were not copied
into the repository.
