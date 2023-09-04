#!/bin/bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

# baseline experiments
python3 main.py --data-set=fmnist --dl-weight=0.0 --logic=dl2 --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=0.0 --logic=dl2 --epochs=200
python3 main.py --data-set=gtsrb --dl-weight=0.0 --logic=dl2 --epochs=200

# Fashion-MNIST Class-Similarity constraint
python3 main.py --data-set=fmnist --dl-weight=0.6 --logic=dl2 --epochs=200
python3 main.py --data-set=fmnist --dl-weight=3.0 --logic=g --epochs=200
python3 main.py --data-set=fmnist --dl-weight=0.8 --logic=kd --epochs=200
python3 main.py --data-set=fmnist --dl-weight=4.0 --logic=lk --epochs=200
python3 main.py --data-set=fmnist --dl-weight=3.0 --logic=gg --epochs=200
python3 main.py --data-set=fmnist --dl-weight=0.8 --logic=rc --epochs=200
python3 main.py --data-set=fmnist --dl-weight=0.8 --logic=rc-s --epochs=200
python3 main.py --data-set=fmnist --dl-weight=1.0 --logic=rc-phi --epochs=200
python3 main.py --data-set=fmnist --dl-weight=1.0 --logic=yg --epochs=200

# CIFAR-10 Class-Similarity constraint
python3 main.py --data-set=cifar10 --dl-weight=0.4 --logic=dl2 --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=1.2 --logic=g --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=0.6 --logic=kd --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=6.0 --logic=lk --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=10.0 --logic=gg --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=0.8 --logic=rc --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=0.8 --logic=rc-s --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=1.6 --logic=rc-phi --epochs=200
python3 main.py --data-set=cifar10 --dl-weight=0.8 --logic=yg --epochs=200

# GTSRB Group constraint
python3 main.py --data-set=gtsrb --dl-weight=7.0 --logic=dl2 --epochs=200
python3 main.py --data-set=gtsrb --dl-weight=5.0 --logic=g --epochs=200
python3 main.py --data-set=gtsrb --dl-weight=5.0 --logic=lk --epochs=200
python3 main.py --data-set=gtsrb --dl-weight=5.0 --logic=rc --epochs=200
python3 main.py --data-set=gtsrb --dl-weight=5.0 --logic=yg --epochs=200

python tables.py
latexmk -pdf -quiet tables.tex
latexmk -pdf -quiet result-plots.tex