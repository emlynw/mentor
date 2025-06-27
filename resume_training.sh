CHECKPOINT_PATH=/home/emlyn/rl_franka/mentor/exp_local/2025.06.23/164104

python train_states_dual_cam_sd_vae.py --config-path $CHECKPOINT_PATH/.hydra --config-name config hydra.run.dir=$CHECKPOINT_PATH