| Property        | Decision          | Why                                                              |
| --------------- | ----------------- | ---------------------------------------------------------------- |
| Model           | Single-layer LSTM | Smallest model that captures temporal dependencies               |
| Input           | 30 × 24           | Sliding window already validated                                 |
| Hidden Size     | 128               | Sufficient latent capacity without excessive parameters          |
| Layers          | 1                 | Baseline before increasing model depth                           |
| Bidirectional   | No                | Future information unavailable during inference                  |
| Dropout         | None              | Single layer; avoid unnecessary regularization                   |
| Regression Head | 128 → 64 → 1      | Small nonlinear mapping from latent representation to scalar RUL |
| Loss            | MSELoss           | Standard regression objective                                    |
| Optimizer       | Adam              | Adaptive optimization for recurrent networks                     |
| Learning Rate   | 1e-3              | Standard Adam baseline                                           |
| Batch Size      | 64                | Already chosen and justified                                     |
| Epochs          | 20 (initial)      | Verify convergence before longer training                        |


EXP-001
Best validation RMSE: 14.67 cycles
Final epoch RMSE:     14.85 cycles
Train RMSE:           12.45 cycles
Epochs:               15
Status:               Baseline successfully trained


Observed behavior:

✓ Correct healthy-region plateau
✓ Correct approximate degradation onset
✓ Correct overall downward degradation trend
✓ Prediction approaches failure region
⚠ Prediction exhibits noticeable timestep-level noise
⚠ Some deviation around degradation onset

Final Verdict: 

Model successfully generalizes to unseen engines without catastrophic overfitting (Train-Test variance < 2.5 cycles).