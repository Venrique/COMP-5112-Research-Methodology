# IAM CNN-LSTM-CTC: final run summary

**Training is finished.** The 20-epoch maximum ended through early stopping
at epoch 17. Epochs 15-17 did not improve the selected score.

| Item | Result |
| --- | --- |
| Selected checkpoint | `best.pt`, epoch 14 |
| Validation CER | 10.07% |
| Validation WER | 34.29% |
| Exact normalized text lines | 11 / 100 |
| Training images per epoch | 6,482 |
| Fixed validation images | 100 |
| Last checkpoint | `last.pt`, epoch 17 |
| Epochs documented | 1-17; no epochs 18-20 |
| Final test evaluation | Not performed |
| Optional preprocessing experiments | Not performed for this model |

Epoch 14 is selected by CER first. Epoch 13 has the lowest WER (34.19%) and
epoch 16 the lowest validation loss (0.3733). Do not call epoch 14 best on every
metric. The latest weights are not the selected weights.

## Open the evidence

- [Complete implementation/methodology/results report](../../../../../docs/CNN_LSTM_CTC.md)
- [Every epoch, with fixed prediction examples](EPOCH_RESULTS.md)
- [Full-precision epoch summary](epoch_summary.csv)
- [Original history](history.csv)
- [Loss/error graphs](training_curves.png)
- [Original epoch 14 predictions](validation_epoch_014.csv)
- [100 epoch 14 examples in random order](examples/epoch_014_random_100_seed5112.md)
- [100-example CSV](examples/epoch_014_random_100_seed5112.csv)
- [Random-order manifest](examples/epoch_014_random_100_seed5112.json)
- [Final verification](FINAL_VERIFICATION.json)
- [Final training status](status.json)
- [Original signed settings](metadata.json)

The random-order export contains all the same 100 validation lines, shuffled
with seed 5112 without replacement. It is not a new evaluation sample.

Weights for epochs 2-17 are archived under `epoch_checkpoints/`. Epoch 1's old
weights were overwritten before archiving began; its metrics/predictions remain.
The selected `best.pt` is included in Git for teammate inference. Other binary
`.pt` files remain Git-ignored: preserve them separately with recorded hashes.

The run is early-stopped with patience counter 3. Increasing the epoch limit
alone does not reset early stopping or start epoch 18. No further training was
started. Next: freeze epoch 14, select optional preprocessing on validation,
and reserve the fixed 250 IAM test lines for final comparison with TrOCR.
