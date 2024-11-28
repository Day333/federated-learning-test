# epoch 10
python main_fed.py --dataset mnist --num_channels 1 --model mlp --epochs 10 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 10 --gpu 0
python main_fed.py --dataset mnist --num_channels 1 --model cnn --epochs 10 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 10 --gpu 0
python main_fed.py --dataset mnist --num_channels 1 --model lr --epochs 10 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model lr --epochs 10 --gpu 0


python main_fed.py --dataset cifar --num_channels 1 --model mlp --epochs 10 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 1 --model mlp --epochs 10 --gpu 0
python main_fed.py --dataset cifar --num_channels 3 --model cnn --epochs 10 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 10 --gpu 0
python main_fed.py --dataset cifar --num_channels 1 --model lr --epochs 10 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 1 --model lr --epochs 10 --gpu 0

# epoch 50

python main_fed.py --dataset mnist --num_channels 1 --model mlp --epochs 50 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 50 --gpu 0
python main_fed.py --dataset mnist --num_channels 1 --model cnn --epochs 50 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0
python main_fed.py --dataset mnist --num_channels 1 --model lr --epochs 50 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model lr --epochs 50 --gpu 0


python main_fed.py --dataset cifar --num_channels 1 --model mlp --epochs 50 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 1 --model mlp --epochs 50 --gpu 0
python main_fed.py --dataset cifar --num_channels 3 --model cnn --epochs 50 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0
python main_fed.py --dataset cifar --num_channels 1 --model lr --epochs 50 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 1 --model lr --epochs 50 --gpu 0


# epoch 100

python main_fed.py --dataset mnist --num_channels 1 --model mlp --epochs 100 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 100 --gpu 0
python main_fed.py --dataset mnist --num_channels 1 --model cnn --epochs 100 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 100 --gpu 0
python main_fed.py --dataset mnist --num_channels 1 --model lr --epochs 100 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model lr --epochs 100 --gpu 0


python main_fed.py --dataset cifar --num_channels 1 --model mlp --epochs 100 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 1 --model mlp --epochs 100 --gpu 0
python main_fed.py --dataset cifar --num_channels 3 --model cnn --epochs 100 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 100 --gpu 0
python main_fed.py --dataset cifar --num_channels 1 --model lr --epochs 100 --gpu 0
python main_fed.py --dataset cifar --iid --num_channels 1 --model lr --epochs 100 --gpu 0