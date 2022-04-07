CUDA_VISIBLE_DEVICES=0 python main.py --num_layers 10 --lr 0.001 --name trial1-lr0.005
CUDA_VISIBLE_DEVICES=0 python main.py --num_layers 10 --lr 0.005 --name trial1-lr0.001
CUDA_VISIBLE_DEVICES=0 python main.py --num_layers 10 --lr 0.0001 --name trial1-lr0.0001
CUDA_VISIBLE_DEVICES=0 python main.py --num_layers 10 --lr 0.00001 --name trial1-lr0.00001