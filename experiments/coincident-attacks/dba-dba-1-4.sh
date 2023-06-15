python simulated_averaging.py \
--lr 0.01 \
--gamma 0.998 \
--num_nets 500 \
--batch-size 64 \
--fl_round 1500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 10 \
--dataset emnist \
--model lenet \
--fl_mode fixed-freq \
--attack_freq 5 \
--attacker_pool_size 100 \
--defense_method no-defense \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--fraction 0.15 \
--adv_lr 0.005 \
--prox_attack False \
--poison_type ardis \
--pdr 0.0 \
--norm_bound 2 \
--ba_strategy dba \
--instance freq-5--benchmark-aba-aba-1-4__same__triger \
--scale_factor 1.0 \
--wandb_group Coincident-backdoor-attacks \
--device=cuda