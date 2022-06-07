# Code for Satellite Attitude Prediction

- Version 1: 2022/4/8

> 1627577388568: Timestamp('2021-07-29 16:49:48.568000')
> 1627548600000

## TODO

- [x] Data process code (cls)
- [x] Training code (cls)
- [ ] Data process code (pred)
- [ ] Training code (pred)
- [ ] Add transformer (pred)

## Result

20220606:

分类
Y：0.9874054789543152
sample: 0.9995706240336101
out_12W: 0.9994268831999406
out_25W: 0.9998832370923914
gf: 0.9999893658927509

回归（lstm）
Y：0.7838753962862319
sample: 0.9999876898877761
out_12W: 0.997205797132555
out_25W: 0.9861617874313187
gf: 0.9950738488099514

回归（transformer）
Y：0.7669140625
sample: 0.9999876898877761
out_12W: 0.997563959478022
out_25W: 0.9969361692994505
gf: 0.9985261927459838
