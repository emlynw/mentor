CHECKPOINT_PATH=/home/emlyn/rl_franka/mentor/exp_local/2025.06.02/143516

python train_adroit.py --config-path $CHECKPOINT_PATH/.hydra --config-name config hydra.run.dir=$CHECKPOINT_PATH