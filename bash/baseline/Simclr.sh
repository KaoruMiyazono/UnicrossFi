CUDA_LAUNCH_BLOCKING=1 python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "SimCLR" \
    --cross_domain_type "location" \
    --target_domain "location1" \
    --device "cuda:0" \
    --num_classes 6 \
    --batch_size 128 \
    --early_stop_epoch 25 \
    --lr 0.0020249102285667528 \
    --weight_decay 0.000735271377290836 \
    --scale_cap 10 \
    --temperature 0.07 \
    --ratio 0.1 \
    --epoch 80 \
    --backbone "ResNet" \
    --seed 42




# python3 /opt/data/private/FDAARC/run_uda.py \
#     --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
#     --csidataset "Widar3.0" \
#     --method "SimCLR" \
#     --cross_domain_type "location" \
#     --target_domain "location1" \
#     --device "cuda:0" \
#     --num_classes 6 \
#     --batch_size 128 \
#     --early_stop_epoch 25 \
#     --lr 0.0020249102285667528 \
#     --weight_decay 0.000735271377290836 \
#     --scale_cap 10 \
#     --temperature 0.07 \
#     --ratio 0.3 \
#     --epoch 80 \
#     --backbone "ResNet" \
#     --seed 42



# python3 /opt/data/private/FDAARC/run_uda.py \
#     --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
#     --csidataset "Widar3.0" \
#     --method "SimCLR" \
#     --cross_domain_type "location" \
#     --target_domain "location1" \
#     --device "cuda:0" \
#     --num_classes 6 \
#     --batch_size 128 \
#     --early_stop_epoch 25 \
#     --lr 0.0020249102285667528 \
#     --weight_decay 0.000735271377290836 \
#     --scale_cap 10 \
#     --temperature 0.07 \
#     --ratio 0.5 \
#     --epoch 80 \
#     --backbone "ResNet" \
#     --seed 42




# python3 /opt/data/private/FDAARC/run_uda.py \
#     --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
#     --csidataset "Widar3.0" \
#     --method "SimCLR" \
#     --cross_domain_type "location" \
#     --target_domain "location1" \
#     --device "cuda:0" \
#     --num_classes 6 \
#     --batch_size 128 \
#     --early_stop_epoch 25 \
#     --lr 0.0020249102285667528 \
#     --weight_decay 0.000735271377290836 \
#     --scale_cap 10 \
#     --temperature 0.07 \
#     --ratio 0.7 \
#     --epoch 80 \
#     --backbone "ResNet" \
#     --seed 42

# python3 /opt/data/private/FDAARC/run_uda.py \
#     --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
#     --csidataset "Widar3.0" \
#     --method "SimCLR" \
#     --cross_domain_type "location" \
#     --target_domain "location1" \
#     --device "cuda:0" \
#     --num_classes 6 \
#     --batch_size 128 \
#     --early_stop_epoch 25 \
#     --lr 0.0020249102285667528 \
#     --weight_decay 0.000735271377290836 \
#     --scale_cap 10 \
#     --temperature 0.07 \
#     --ratio 0.9 \
#     --epoch 80 \
    #   --backbone "ResNet" \
    #   --seed 42


python3 /opt/data/private/FDAARC/run_uda.py \
    --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" \
    --csidataset "Widar3.0" \
    --method "SimCLR" \
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
    --ratio 0.1 \
    --backbone "ResNet" \
    --seed 42