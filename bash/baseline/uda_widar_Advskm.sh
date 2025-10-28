
# lr 0.000273126963327332187  weight_decay 0.006304033121728342 src_lossweight 7.14301251023217134 domain_loss_weight 0.82209085732910147
# tdcsi lr: 0.006498551297771579 weight_decay 0.00000118027204615218  src_loss_weight 0.5350675723495651 domain_loss 2.282130922931133
# tdcsi_2_a lr 0.007873520974854817 weight_decay 0.00000572251437988261 src_loss_weight 0.9535698332081244 domain_loss 6.969513346296931
# resnet2d lr 0.008538540813867289 weight_decay 0.00017889769269179153 src_loss_weight  1.4350189349258753 domain_loss  0.09171043903675574


# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3 --backbone 'ResNet'






# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room2" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room2" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room2" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room2" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3 --backbone 'ResNet'





# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1  --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2  --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "room" --target_domain "room3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3  --backbone 'ResNet'



# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2  --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3 --backbone 'ResNet'





# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3 --backbone 'ResNet'




# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location5" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location5" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location5" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "location" --target_domain "location5" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3 --backbone 'ResNet'


# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3 --backbone 'ResNet'




# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3 --backbone 'ResNet'




# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user7" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user7" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user7" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "user" --target_domain "user7" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 3 --backbone 'ResNet'

 



# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
# python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation1" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'


python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation2" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation2" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation2" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'


python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 0 --backbone 'ResNet'
python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 1 --backbone 'ResNet'
python3 run_uda.py --data_dir "/opt/data/private/ablation_study/data_widar_800/tdcsi/" --csidataset "Widar3.0" --method "AdvSKM" --inputshap 60 800 --cross_domain_type "orientation" --target_domain "orientation3" --device "cuda:0" --lr 0.008538540813867289 --weight_decay 0.00017889769269179153 --src_loss_weight 1.4350189349258753 --domain_loss_weight 0.09171043903675574 --seed 2 --backbone 'ResNet'