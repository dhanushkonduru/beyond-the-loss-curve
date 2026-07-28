# Datasets

All records originate from the World Health Organization Global Health Observatory (<https://www.who.int/data/gho>), retrieved through the public API at `https://ghoapi.azureedge.net/api`. Nothing here is synthetic.

## `who_benchmark/`

The hundred-entity benchmark analysed in Sections VI and VII of the manuscript, split into its three disjoint partitions:

| File | Role |
|---|---|
| `entities.json` | The 100 source entities, five fields each |
| `full.json` | All 2000 question–answer pairs (20 per entity) |
| `forget.json` | The 20-entity deletion request (400 pairs) |
| `retain.json` | The 80-entity set that must survive (1600 pairs) — anchors the unlearning update and supplies the utility control group |
| `heldout.json` | 20 entities never trained on — supplies genuine non-members for the membership inference attack |

## `who_sync_releases/`

Two successive annual WHO releases plus their computed diff, used for the live dataset–model synchronisation cycle reported in Section VII-F (Table 5): one cycle learns the added records, unlearns the withdrawn records, and preserves the rest — without retraining.

| File | Role |
|---|---|
| `release_2023.json` / `release_2024.json` | The two dataset releases, as question–answer pairs |
| `records_2023.json` / `records_2024.json` | The same two releases as structured entity records |
| `diff.json` | The precomputed added / removed / kept partition between the releases |

## Regenerating instead of transferring

Construction is deterministic given the source release and the stated random seed (see the manuscript's data availability statement and Section VI-C2), so this benchmark can be rebuilt exactly with [`../2_Source code/pipeline/generate_dataset.py`](../2_Source%20code/pipeline/generate_dataset.py) and [`../2_Source code/pipeline/sync_fetch.py`](../2_Source%20code/pipeline/sync_fetch.py) rather than only transferred as static files.
