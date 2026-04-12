# Wandb / Optuna Tuning Summary

## Wandb
- A first wandb-based tuning workflow was set up and tested.
- Small controlled runs were logged successfully.

## Optuna
- Number of trials: 8
- Best value: 2.628667e-05
- Best params: {'lr': 0.00015917718036895313, 'base_ch': 32, 'batch_size': 2}

## Final selected run
- run_name: `optuna_best_lr1.6e-04_base32_bs2_ep50`
- lr: `0.00015917718036895313`
- base_ch: `32`
- batch_size: `2`
- epochs: `50`
- final summary: {'best_epoch': 44, 'best_val_loss': 1.097921534346824e-05, 'final_val_mse': 1.3778796175756724e-05, 'final_val_mae': 0.002420939379371703, 'elapsed_min': 10.127114061514536, 'best_ckpt_path': '/content/drive/MyDrive/rectified_flow_ct2dose/outputs/checkpoints/optuna_best_lr1.6e-04_base32_bs2_ep50_best.pt', 'latest_ckpt_path': '/content/drive/MyDrive/rectified_flow_ct2dose/outputs/checkpoints/optuna_best_lr1.6e-04_base32_bs2_ep50_latest.pt'}