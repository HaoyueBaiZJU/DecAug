echo ERM NICO animal
CUDA_VISIBLE_DEVICE=0 python ERM_main.py \
--nico_cls animal \
--epochs 30 \
--num_classes 10 \
--batch_size 128 \
--learning_rate 0.0005 \
--weight_decay 0.0005 \
--data_root /home/ma-user/work/OOD/data/nico/ \

echo ERM NICO vehicle
CUDA_VISIBLE_DEVICE=0 python ERM_main.py \
--nico_cls vehicle \
--epochs 30 \
--num_classes 10 \
--batch_size 128 \
--learning_rate 0.0005 \
--weight_decay 0.0005 \
--data_root /home/ma-user/work/OOD/data/nico/ \