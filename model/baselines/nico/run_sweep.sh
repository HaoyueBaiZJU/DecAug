algorithm="MLDG"
echo $algorithm
nohup python train_nico.py --algorithm $algorithm --gpu 1 --nico_cls animal >>results/res_$algorithm"_animal" &
nohup python train_nico.py --algorithm $algorithm --gpu 2 --nico_cls vehicle >>results/res_$algorithm"_vehicle" &
algorithm="MMD"
echo $algorithm
nohup python train_nico.py --algorithm $algorithm --gpu 3 --nico_cls animal >>results/res_$algorithm"_animal" &
nohup python train_nico.py --algorithm $algorithm --gpu 0 --nico_cls vehicle >>results/res_$algorithm"_vehicle" &
