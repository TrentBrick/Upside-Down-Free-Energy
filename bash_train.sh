parallel python trainer.py --gamename lunarlander \
--exp_name levine_cum_rew_minus_std \
--num_workers 1 --no_reload --seed {1} ::: {25..29}
# inclusive numbers