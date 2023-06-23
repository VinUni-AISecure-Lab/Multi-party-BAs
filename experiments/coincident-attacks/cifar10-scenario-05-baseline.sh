python simulated_averaging.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 10 \
--dataset cifar10 \
--batch-size 64 \
--model vgg9 \
--attack_freq 5 \
--fl_mode fixed-freq \
--attacker_pool_size 100 \
--defense_method no-defense \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--single_attack True \
--stddev 0.025 \
--scenario_idx 5 \
--eps 2 \
--adv_lr 0.01 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--ba_strategy dba \
--instance freq5__benchmark__scenario__05__CIFAR10 \
--scale_factor 1.0 \
--wandb_group Coincident-backdoor-attacks \
--device=cuda