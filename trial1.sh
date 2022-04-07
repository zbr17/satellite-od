CUDA_VISIBLE_DEVICES=1 python main.py --num_layers 1 --lr 0.001 --name trial1-layer1
CUDA_VISIBLE_DEVICES=1 python main.py --num_layers 2 --lr 0.001 --name trial1-layer2
CUDA_VISIBLE_DEVICES=1 python main.py --num_layers 3 --lr 0.001 --name trial1-layer3
CUDA_VISIBLE_DEVICES=1 python main.py --num_layers 5 --lr 0.001 --name trial1-layer5
CUDA_VISIBLE_DEVICES=1 python main.py --num_layers 10 --lr 0.001 --name trial1-layer10
CUDA_VISIBLE_DEVICES=1 python main.py --num_layers 20 --lr 0.001 --name trial1-layer20