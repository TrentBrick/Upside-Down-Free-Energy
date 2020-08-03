for i in 25 26 27 28 29
do 
    mkdir exp_dir/lunarlander-sparse/levine_init_hparams/seed_$i
    mv exp_dir/lunarlander-sparse/seed_$i/logger/version_5/ exp_dir/lunarlander-sparse/levine_init_hparams/seed_$i
done