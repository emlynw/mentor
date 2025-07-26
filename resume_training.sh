CHECKPOINT_PATH=/home/emlyn/rl_franka/mentor/exp_local/2025.07.23/095446_

python train_asym_dual_cam_sd_vae.py --config-path $CHECKPOINT_PATH/.hydra --config-name config hydra.run.dir=$CHECKPOINT_PATH