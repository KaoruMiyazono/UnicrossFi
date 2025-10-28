# room 
python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "room" \
    --target_domain "room1" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.003 \
    --weight_decay 3e-4 \
    --scale_cap 6 \
    --temperature 0.07 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "room" \
    --target_domain "room2" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0047 \
    --weight_decay 0.00016 \
    --scale_cap 6 \
    --temperature 0.07 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "room" \
    --target_domain "room3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.003 \
    --weight_decay 3e-4 \
    --scale_cap 6 \
    --temperature 0.1069 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

# location
python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "location" \
    --target_domain "location1" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.002 \
    --weight_decay 0.0007 \
    --scale_cap 6 \
    --temperature 0.07 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "location" \
    --target_domain "location3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.002 \
    --weight_decay 0.0007 \
    --scale_cap 6 \
    --temperature 0.07 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "location" \
    --target_domain "location5" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.002 \
    --weight_decay 0.0004 \
    --scale_cap 6 \
    --temperature 0.098 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

#user
python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "user" \
    --target_domain "user1" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.002 \
    --weight_decay 0.0002 \
    --scale_cap 6 \
    --temperature 0.1615 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "user" \
    --target_domain "user3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.004 \
    --weight_decay 0.00055 \
    --scale_cap 6 \
    --temperature 0.07 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "user" \
    --target_domain "user7" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0015 \
    --weight_decay 0.00012 \
    --scale_cap 6 \
    --temperature 0.07 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

# orientation 
python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation1" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation1" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 1994

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation1" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 2025





python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation2" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation2" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 1994

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation2" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 2025



python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 1994

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_dg" \
    --cross_domain_type "orientation" \
    --target_domain "orientation3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0014458690230267932 \
    --weight_decay 0.00036358891036611363 \
    --scale_cap 6 \
    --temperature 0.11 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 2025

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_semidg" \
    --cross_domain_type "location" \
    --target_domain "location3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0020249102285667528 \
    --weight_decay 0.0007352713772908367 \
    --scale_cap 6 \
    --temperature 0.0706 \
    --epoch 80 \
    --backbone "ResNet" \
    --ratio 0.1 \
    --seed 42 \
    --pseudo_label_threshold 0.75

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_semidg" \
    --cross_domain_type "location" \
    --target_domain "location3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0020249102285667528 \
    --weight_decay 0.0007352713772908367 \
    --scale_cap 6 \
    --temperature 0.0706 \
    --epoch 80 \
    --backbone "ResNet" \
    --ratio 0.3 \
    --seed 42 \
    --pseudo_label_threshold 0.75
python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_semidg" \
    --cross_domain_type "location" \
    --target_domain "location3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0020249102285667528 \
    --weight_decay 0.0007352713772908367 \
    --scale_cap 6 \
    --temperature 0.0706 \
    --epoch 80 \
    --backbone "ResNet" \
    --ratio 0.5 \
    --seed 42 \
    --pseudo_label_threshold 0.75

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_semidg" \
    --cross_domain_type "location" \
    --target_domain "location3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0020249102285667528 \
    --weight_decay 0.0007352713772908367 \
    --scale_cap 6 \
    --temperature 0.0706 \
    --epoch 80 \
    --backbone "ResNet" \
    --ratio 0.7 \
    --seed 42 \
    --pseudo_label_threshold 0.75

python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "UniCrossFi_semidg" \
    --cross_domain_type "location" \
    --target_domain "location3" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0020249102285667528 \
    --weight_decay 0.0007352713772908367 \
    --scale_cap 6 \
    --temperature 0.0706 \
    --epoch 80 \
    --backbone "ResNet" \
    --ratio 0.9 \
    --seed 42 \
    --pseudo_label_threshold 0.75