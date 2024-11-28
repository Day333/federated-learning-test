# number_client_frac
python main_fed.py --dataset cifar --num_channels 1 --model mlp --epochs 50 --gpu 0 
python main_fed.py --dataset cifar --iid --num_channels 1 --model mlp --epochs 50 --gpu 0 
python main_fed.py --dataset cifar --num_channels 3 --model cnn --epochs 50 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0
python main_fed.py --dataset cifar --num_channels 1 --model lr --epochs 50 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 1 --model lr --epochs 50 --gpu 0

python main_fed.py --dataset cifar --num_channels 1 --model mlp --epochs 50 --gpu 0 --frac 0.5
python main_fed.py --dataset cifar --iid --num_channels 1 --model mlp --epochs 50 --gpu 0 --frac 0.5
python main_fed.py --dataset cifar --num_channels 3 --model cnn --epochs 50 --gpu 0 --frac 0.5
python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0 --frac 0.5
python main_fed.py --dataset cifar --num_channels 1 --model lr --epochs 50 --gpu 0 --frac 0.5
python main_fed.py --dataset cifar --iid --num_channels 1 --model lr --epochs 50 --gpu 0 --frac 0.5

python main_fed.py --dataset cifar --num_channels 1 --model mlp --epochs 50 --gpu 0 --all_clients
python main_fed.py --dataset cifar --iid --num_channels 1 --model mlp --epochs 50 --gpu 0 --all_clients
python main_fed.py --dataset cifar --num_channels 3 --model cnn --epochs 50 --gpu 0 --all_clients
python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0 --all_clients
python main_fed.py --dataset cifar --num_channels 1 --model lr --epochs 50 --gpu 0 --all_clients
python main_fed.py --dataset cifar --iid --num_channels 1 --model lr --epochs 50 --gpu 0 --all_clients