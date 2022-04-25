import os

for i in range(1, 10):
    os.system("python train.py --cfg configs/phyre/rptran_within_pred%d.yaml --gpus 0 --output within_pred/%d --seed 0" % (i, i))


