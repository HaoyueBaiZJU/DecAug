algorithm="ERM"
echo $algorithm

#nohup python train_pacs_single.py --algorithm $algorithm --gpu 1 --source cartoon --target cartoon >>res_single/res_$algorithm"_cartoon" & 
#nohup python train_pacs_single.py --algorithm $algorithm --gpu 2 --source sketch --target sketch >>res_single/res_$algorithm"_sketch" &
#nohup python train_pacs_single.py --algorithm $algorithm --gpu 3 --source photo --target photo >>res_single/res_$algorithm"_photo" &
nohup python train_pacs_single.py --algorithm $algorithm --gpu 0 --source art_painting --target art_painting >>res_single/res_$algorithm"_painting" &
