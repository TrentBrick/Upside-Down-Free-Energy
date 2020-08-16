parallel python trainer.py --gamename lunarlander \
--exp_name levine_rew_to_go_no_time_limit \
--num_workers 1 --no_reload --seed {1} ::: {25..27}
# inclusive numbers