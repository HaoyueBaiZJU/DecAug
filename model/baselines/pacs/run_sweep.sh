algorithm="ERM"
echo $algorithm

nohup python train_pacs.py --algorithm $algorithm --gpu 1 --source sketch photo art_painting --target cartoon >>res_no_pretrain/res_$algorithm"_cartoon" & 
nohup python train_pacs.py --algorithm $algorithm --gpu 2 --source cartoon photo art_painting --target sketch >>res_no_pretrain/res_$algorithm"_sketch" &
nohup python train_pacs.py --algorithm $algorithm --gpu 3 --source sketch cartoon art_painting --target photo >>res_no_pretrain/res_$algorithm"_photo" &
nohup python train_pacs.py --algorithm $algorithm --gpu 0 --source sketch photo cartoon --target art_painting >>res_no_pretrain/res_$algorithm"_painting" &
