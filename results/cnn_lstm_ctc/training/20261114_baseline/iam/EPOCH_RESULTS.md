# CNN + LSTM + CTC: epoch results

Generated from recorded history and validation predictions. Keep manual discussion in a separate file; this report refreshes automatically.

- Dataset: **IAM**; original training split, no optional preprocessing.
- Training lines: **6,482**. Fixed validation lines: **100**.
- Requested maximum: **20 epochs**; **17 completed**. Early stopping still applies.
- Dataset revision: `fbdad97500ce54635c0d1ba306bf535cb40656cf`.
- Seed: `5112`; full model, vocabulary, source fingerprints, software versions, and original settings: [metadata.json](metadata.json).
- Scores are **validation results**, not final test results. Corpus CER/WER are recomputed from each saved prediction CSV as a consistency check.
- Scoring collapses whitespace and preserves case/punctuation. Lower loss, CER, and WER are better. CER is an edit rate, not a simple accuracy percentage.
- Training loss is averaged while weights change; validation loss is measured afterward with fixed weights. Loss alone does not establish reading quality.
- Training time is elapsed time and can include standby pauses; do not treat it as uninterrupted compute time.

## Completed epochs

| Epoch | Train loss | Validation loss | CER (%) | WER (%) | CER change (pp) | WER change (pp) | Exact lines | Empty outputs | Train min | Predictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.2313 | 3.1311 | 97.28 | 98.82 | - | - | 0/100 | 0 | 28.9 | [CSV](validation_epoch_001.csv) |
| 2 | 2.6515 | 2.1477 | 60.53 | 91.35 | -36.75 | -7.48 | 0/100 | 0 | 93.5 | [CSV](validation_epoch_002.csv) |
| 3 | 1.5982 | 1.1219 | 32.34 | 71.47 | -28.20 | -19.87 | 0/100 | 0 | 22.9 | [CSV](validation_epoch_003.csv) |
| 4 | 1.0281 | 0.7907 | 23.31 | 62.07 | -9.03 | -9.40 | 0/100 | 0 | 20.8 | [CSV](validation_epoch_004.csv) |
| 5 | 0.7745 | 0.6258 | 17.76 | 52.24 | -5.54 | -9.83 | 2/100 | 0 | 20.5 | [CSV](validation_epoch_005.csv) |
| 6 | 0.6322 | 0.5305 | 15.34 | 47.86 | -2.42 | -4.38 | 2/100 | 0 | 20.8 | [CSV](validation_epoch_006.csv) |
| 7 | 0.5348 | 0.5007 | 14.57 | 45.09 | -0.77 | -2.78 | 4/100 | 0 | 20.8 | [CSV](validation_epoch_007.csv) |
| 8 | 0.4675 | 0.4544 | 12.58 | 41.56 | -1.99 | -3.53 | 7/100 | 0 | 21.0 | [CSV](validation_epoch_008.csv) |
| 9 | 0.4161 | 0.4639 | 12.97 | 41.56 | +0.38 | +0.00 | 5/100 | 0 | 20.5 | [CSV](validation_epoch_009.csv) |
| 10 | 0.3756 | 0.4146 | 11.77 | 38.68 | -1.20 | -2.88 | 10/100 | 0 | 21.4 | [CSV](validation_epoch_010.csv) |
| 11 | 0.3387 | 0.4036 | 10.86 | 36.54 | -0.91 | -2.14 | 4/100 | 0 | 20.7 | [CSV](validation_epoch_011.csv) |
| 12 | 0.3107 | 0.4067 | 11.22 | 36.75 | +0.36 | +0.21 | 9/100 | 0 | 21.1 | [CSV](validation_epoch_012.csv) |
| 13 | 0.2857 | 0.3802 | 10.45 | 34.19 | -0.77 | -2.56 | 10/100 | 0 | 31.5 | [CSV](validation_epoch_013.csv) |
| 14 | 0.2595 | 0.3757 | 10.07 | 34.29 | -0.38 | +0.11 | 11/100 | 0 | 36.1 | [CSV](validation_epoch_014.csv) |
| 15 | 0.2422 | 0.4187 | 10.64 | 36.22 | +0.57 | +1.92 | 6/100 | 0 | 28.6 | [CSV](validation_epoch_015.csv) |
| 16 | 0.2228 | 0.3733 | 10.30 | 35.36 | -0.34 | -0.85 | 8/100 | 0 | 21.9 | [CSV](validation_epoch_016.csv) |
| 17 | 0.2073 | 0.4068 | 10.61 | 36.54 | +0.32 | +1.18 | 8/100 | 0 | 21.1 | [CSV](validation_epoch_017.csv) |

Changes are versus the previous epoch in percentage points (pp); negative means fewer errors.

**Best completed epoch: 14**, selected by validation CER, then WER, then validation loss. This can differ from the latest epoch.

[Full-precision epoch table](epoch_summary.csv) · [Original history](history.csv) · [Loss/error graphs](training_curves.png)

## Predictions and checkpoint notes by epoch

The same first validation sample IDs are shown each epoch; these are fixed examples, not chosen to look successful.

### Epoch 1

- Validation CER: **97.28%**; WER: **98.82%**.
- Exact normalized text lines: **0/100**; empty predictions: **0**.
- No separate weights archive is available for this epoch: archiving began later. Its original scores and prediction CSV are preserved; weights have not been reconstructed.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | ts |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | ts |
| sample_27 | Most of those who expound a theory of textual | s |

### Epoch 2

- Validation CER: **60.53%**; WER: **91.35%**.
- Exact normalized text lines: **0/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_002.pt](epoch_checkpoints/epoch_002.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | bede wd afee te the p i t |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | ay tg wed wthl t thes the plees hte a ty |
| sample_27 | Most of those who expound a theory of textual | ct ot theee ahe apd a tet o tedt |

### Epoch 3

- Validation CER: **32.34%**; WER: **71.47%**.
- Exact normalized text lines: **0/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_003.pt](epoch_checkpoints/epoch_003.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | mroatinme manl ofeee in the Bp ii |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | af h mat maneeatiy I hr the pulaeeo ito a htny |
| sample_27 | Most of those who expound a theory of textual | Mast of thase whs eponnd a theoy of tetal |

### Epoch 4

- Validation CER: **23.31%**; WER: **62.07%**.
- Exact normalized text lines: **0/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_004.pt](epoch_checkpoints/epoch_004.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | roatie maval offece in the bgc iin |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at h mat naueeatiy ) tuns the palaelo ito a laxuy |
| sample_27 | Most of those who expound a theory of textual | Host of those whs erponnd a thesy of tetna , |

### Epoch 5

- Validation CER: **17.76%**; WER: **52.24%**.
- Exact normalized text lines: **2/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_005.pt](epoch_checkpoints/epoch_005.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | mastiime maval offhiece inn the b0h ii |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at h most naussatiy ) tums the palasso into a txury |
| sample_27 | Most of those who expound a theory of textual | Hast of those whs exponnd a thesry of totnal |

### Epoch 6

- Validation CER: **15.34%**; WER: **47.86%**.
- Exact normalized text lines: **2/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_006.pt](epoch_checkpoints/epoch_006.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastiie maval offiece in the B0C ' |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at hs most nauseatiy ) tums the palazlo into a lhreury |
| sample_27 | Most of those who expound a theory of textual | Host of those whe exponnd a thesr of totnal |

### Epoch 7

- Validation CER: **14.57%**; WER: **45.09%**.
- Exact normalized text lines: **4/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_007.pt](epoch_checkpoints/epoch_007.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastime naval offiecr in the D0C 's |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at h most nauseatiny ) tums the palasso into a threury |
| sample_27 | Most of those who expound a theory of textual | Mast of those whs exponnd a thesry of tatnal |

### Epoch 8

- Validation CER: **12.58%**; WER: **41.56%**.
- Exact normalized text lines: **7/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_008.pt](epoch_checkpoints/epoch_008.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastime aval offiecr in the BBC'n |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at hr most nauseatiny ) tums the palazsg into a lureury |
| sample_27 | Most of those who expound a theory of textual | Mast of those whs sxponnd a thesry of tetual |

### Epoch 9

- Validation CER: **12.97%**; WER: **41.56%**.
- Exact normalized text lines: **5/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_009.pt](epoch_checkpoints/epoch_009.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastinme naval offiecr in the BC'n |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at hs most nauseatiny ) tums the palazeo into a Iuxeury |
| sample_27 | Most of those who expound a theory of textual | Hast of those whs exponnd a thesry of totnal |

### Epoch 10

- Validation CER: **11.77%**; WER: **38.68%**.
- Exact normalized text lines: **10/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_010.pt](epoch_checkpoints/epoch_010.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastime noval offieer in the BBC'n |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at t most nauseatig ) tums the palazeo into a luxeury |
| sample_27 | Most of those who expound a theory of textual | Host of those whs exponnd a thesry of tetual |

### Epoch 11

- Validation CER: **10.86%**; WER: **36.54%**.
- Exact normalized text lines: **4/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_011.pt](epoch_checkpoints/epoch_011.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastime naval offiecr in the BBC "' s |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at hs most nauseatiy ) tums the palazlo into a luxeury |
| sample_27 | Most of those who expound a theory of textual | Mast of those whs expound a thesry of tetnal |

### Epoch 12

- Validation CER: **11.22%**; WER: **36.75%**.
- Exact normalized text lines: **9/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_012.pt](epoch_checkpoints/epoch_012.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastime saval offieer in the BBC's |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at h most nauseatilg ) turs the palazto into a lhxcury |
| sample_27 | Most of those who expound a theory of textual | Most of those whs expound a thear of tetnal |

### Epoch 13

- Validation CER: **10.45%**; WER: **34.19%**.
- Exact normalized text lines: **10/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_013.pt](epoch_checkpoints/epoch_013.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastime naval offiecr in the BBC " s |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at hs most nauseatig ) tums the palazso into a lnseury |
| sample_27 | Most of those who expound a theory of textual | Mast of those wha exponnd a thesty of testual |

### Epoch 14

- Validation CER: **10.07%**; WER: **34.29%**.
- Exact normalized text lines: **11/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_014.pt](epoch_checkpoints/epoch_014.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | waitime naval offieer in the BBC's |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at his most nanseating ) tums the palazze into a lhxeury |
| sample_27 | Most of those who expound a theory of textual | Mast of those wha expound a thear of tetnal |

### Epoch 15

- Validation CER: **10.64%**; WER: **36.22%**.
- Exact normalized text lines: **6/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_015.pt](epoch_checkpoints/epoch_015.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | wastime naval offieee in the B0C ' s |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at his most nauseatiy ) turs the palazeo into a Iuxeury |
| sample_27 | Most of those who expound a theory of textual | Most of those wha expound a theary of tetnal |

### Epoch 16

- Validation CER: **10.30%**; WER: **35.36%**.
- Exact normalized text lines: **8/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_016.pt](epoch_checkpoints/epoch_016.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | waitime noval offieer in the B8C'19. |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at his most nauseating ) turs the palazso into a luxeury |
| sample_27 | Most of those who expound a theory of textual | Most of those wha expound a theary of tetnal |

### Epoch 17

- Validation CER: **10.61%**; WER: **36.54%**.
- Exact normalized text lines: **8/100**; empty predictions: **0**.
- Archived weights and training state: [epoch_checkpoints/epoch_017.pt](epoch_checkpoints/epoch_017.pt). SHA-256 is recorded in `epoch_summary.csv`.

| Sample | Correct text | Predicted text |
| --- | --- | --- |
| sample_6 | wartime naval officer in the B B C's | waitime naval officer in the B8C's |
| sample_25 | at his most nauseating ) turns the palazzo into a luxury | at his most nauseatilg ) turs the palazso into a luxeury |
| sample_27 | Most of those who expound a theory of textual | Most of those wha expound a theaty of tatnal |
