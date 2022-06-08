# python data/process.py --data Y
# python data/process.py --data sample
# python data/process.py --data gf
# python data/process.py --data out_12W
# python data/process.py --data out_25W

# python main_pred.py --sample_name Y --model_name lstm --epochs 5
# python main_pred.py --sample_name sample --model_name lstm --epochs 2
# python main_pred.py --sample_name gf --model_name lstm --epochs 1
# python main_pred.py --sample_name out_12W --model_name lstm --epochs 2
# python main_pred.py --sample_name out_25W --model_name lstm --epochs 2

python main_pred.py --sample_name Y --model_name transformer --epochs 5
python main_pred.py --sample_name sample --model_name transformer --epochs 2
python main_pred.py --sample_name gf --model_name transformer --epochs 1
python main_pred.py --sample_name out_12W --model_name transformer --epochs 2
python main_pred.py --sample_name out_25W --model_name transformer --epochs 2

# python main_cls.py --sample_name Y 
# python main_cls.py --sample_name sample 
# python main_cls.py --sample_name gf 
# python main_cls.py --sample_name out_12W 
# python main_cls.py --sample_name out_25W 