parallel python trainer.py --gamename lunarlander \
--num_workers 1 --no_reload --seed {1} --num_grad_steps 10 ::: {20..24}
# inclusive numbers