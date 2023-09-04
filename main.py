from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
import time
import os
import sys
import csv
import functools

from models import MnistNet, GtsrbNet, ResNet18
from constraints import FashionMnistConstraint, Cifar10Constraint, GtsrbConstraint
from logics import *

def logical_loss(x_batch, y_batch, output, logic, constraint, args):
    constraint.set_model_output(output)

    loss, sat = constraint.eval(logic)

    # Fuzzy logics use 1 as absolute truth
    if args.logic == 'dl2':
        return torch.mean(loss), torch.mean(sat.float())
    else:
        return 1.0 - torch.mean(loss), torch.mean(sat.float())

def train(args, model, device, train_loader, optimizer, dl_weight, epoch, logic, constraint):
    avg_pred_acc = torch.tensor(0., device=device)
    avg_pred_loss = torch.tensor(0., device=device)
    avg_constr_acc = torch.tensor(0., device=device)
    avg_constr_loss = torch.tensor(0., device=device)

    steps = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        steps += 1

        x_batch, y_batch = data.to(device), target.to(device)

        output = model(x_batch)
        ce_loss = F.cross_entropy(output, y_batch)

        x_correct = torch.mean(torch.argmax(output, dim=1).eq(y_batch).float())

        if dl_weight > 0.0:
            dl_loss, sat = logical_loss(x_batch, y_batch, output, logic, constraint, args)
            total_loss = ce_loss + dl_weight * dl_loss
        elif dl_weight < 0.0:
            dl_loss, sat = logical_loss(x_batch, y_batch, output, logic, constraint, args)
            total_loss = dl_loss
        else:
            total_loss = ce_loss

        model.train()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if dl_weight != 0.0:
            avg_pred_acc += x_correct
            avg_constr_acc += sat
            avg_pred_loss += ce_loss
            avg_constr_loss += dl_loss
                
    return avg_pred_acc.item() / float(steps), avg_constr_acc.item() / float(steps), avg_pred_loss.item() / float(steps), avg_constr_loss.item() / float(steps)

def test(args, model, device, test_loader, epoch, logic, constraint):
    model.eval()
    test_ce_loss = torch.tensor(0., device=device)
    test_dl_loss = torch.tensor(0., device=device)
    
    correct = torch.tensor(0., device=device)
    constr = torch.tensor(0., device=device)

    steps = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            steps += 1

            data, target = data.to(device), target.to(device)
            output = model(data)

            dl_loss, sat = logical_loss(data, target, output, logic, constraint, args)
            
            test_ce_loss += F.cross_entropy(output, target)
            test_dl_loss += dl_loss
            
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()

            constr += sat

    test_ce_loss /= len(test_loader.dataset)
    test_dl_loss /= len(test_loader.dataset)

    correct = correct.item()
    constr = constr.item()
    test_ce_loss = test_ce_loss.item()
    test_dl_loss = test_dl_loss.item()

    return correct / len(test_loader.dataset), constr / float(steps), test_ce_loss, test_dl_loss

def label_noise(p, num_classes, y):
    if torch.rand(1).item() < p:
        return (y + torch.randint(1, num_classes, (1,)).item()) % num_classes
    else:
        return y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--data-set', type=str, default='fmnist')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--dl-weight', type=float, default=0.1)
    parser.add_argument('--logic', type=str, default='dl2')
    parser.add_argument('--reports-dir', type=str, default='reports')
    parser.add_argument('--lambda-search', type=bool, default=False)
    args = parser.parse_args()

    print(sys.argv)
    
    torch.manual_seed(42)
    np.random.seed(42)   

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print('Using device {}'.format(device))

    kwargs = {'batch_size': args.batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 4, 'pin_memory': True }
        kwargs.update(cuda_kwargs)

    logics = {
        'dl2': DL2(),
        'g': GödelFuzzyLogic(),
        'kd': KleeneDienesFuzzyLogic(),
        'lk': LukasiewiczFuzzyLogic(),
        'rc': ReichenbachFuzzyLogic(),
        'gg': GoguenFuzzyLogic(),
        'rc-s': ReichenbachSigmoidalFuzzyLogic(),
        'rc-phi': ReichenbachBijectionFuzzyLogic(),
        'yg': YagerFuzzyLogic(),
    }

    logic = logics[args.logic]

    constraints = {
        'fmnist': FashionMnistConstraint(torch.tensor(0.1, device=device)),
        'cifar10': Cifar10Constraint(torch.tensor(0.1, device=device)),
        'gtsrb': GtsrbConstraint(torch.tensor(0.03, device=device))
    }

    constraint = constraints[args.data_set]

    # NOTE: the class similarity constraint uses a big outer conjunction, but is not supposed to compare different conjunctions so we use the same for all fuzzy logics.
    #       Similar for the group constraint, which uses inner disjunctions but is not supposed to compare different disjunctions.
    for l in logics.values():
        if isinstance(l, FuzzyLogic):
            if args.data_set in ['fmnist', 'cifar10']:
                l._AND = logics['rc']._AND
            elif args.data_set == 'gtsrb':
                l._OR = logics['rc']._OR
    
    if args.data_set == 'fmnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        fraction = 0.1

        dataset_train = datasets.FashionMNIST('data', train=True, download=True, transform=transform_train, target_transform=functools.partial(label_noise, 0.1, 10))
        dataset_test = datasets.FashionMNIST('data', train=False, transform=transform_test)

        model = MnistNet().to(device)
    elif args.data_set == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])

        fraction = 0.5

        transform_test = transforms.Compose([transforms.ToTensor()])

        dataset_train = datasets.CIFAR10('data', train=True, download=True, transform=transform_train, target_transform=functools.partial(label_noise, 0.1, 10))
        dataset_test = datasets.CIFAR10('data', train=False, transform=transform_test)
        
        model = ResNet18().to(device)
    elif args.data_set == 'gtsrb':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                (0.2724, 0.2608, 0.2669))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                (0.2724, 0.2608, 0.2669))
        ])

        fraction = 0.9

        dataset_train = datasets.GTSRB('data', split="train", download=True, transform=transform_train, target_transform=functools.partial(label_noise, 0.1, 43))
        dataset_test = datasets.GTSRB('data', split="test", download=True, transform=transform_test)
        
        model = GtsrbNet().to(device)

    num_samples = len(dataset_train)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    subset_indices = indices[:int(fraction * num_samples)]

    train_loader = torch.utils.data.DataLoader(Subset(dataset_train, subset_indices), shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, **kwargs)

    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    os.makedirs(args.reports_dir, exist_ok=True)

    if args.lambda_search:
        file_name = f'{args.reports_dir}/report_{args.data_set}_{args.logic if args.dl_weight != 0.0 else "baseline"}_{args.dl_weight}_{args.epochs}.csv'
    else:
        file_name = f'{args.reports_dir}/report_{args.data_set}_{args.logic if args.dl_weight != 0.0 else "baseline"}.csv'

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        csvfile.write(f'#{sys.argv}\n')
        writer.writerow(['Epoch', 'Train-P-Loss', 'Train-C-Loss', 'Train-P-Acc', 'Train-C-Acc', 'Test-P-Loss', 'Test-C-Loss', 'Test-P-Acc', 'Test-C-Acc', 'Time'])

        for epoch in range(0, args.epochs + 1):
            start = time.time()

            if epoch > 0:
                train_pred_acc, train_constr_acc, train_pred_loss, train_constr_loss = train(args, model, device, train_loader, optimizer, args.dl_weight, epoch, logic, constraint)
            else:
                train_pred_acc, train_constr_acc, train_pred_loss, train_constr_loss = 0, 0, 0, 0

            train_time = time.time() - start

            test_pred_acc, test_constr_acc, test_pred_loss, test_constr_loss = test(args, model, device, test_loader, epoch, logic, constraint)

            epoch_time = time.time() - start
        
            writer.writerow([epoch, train_pred_loss, train_constr_loss, train_pred_acc, train_constr_acc, test_pred_loss, test_constr_loss, test_pred_acc, test_constr_acc, train_time])
            
            print(f'Epoch {epoch} / {args.epochs}\t Config: {args.logic if args.dl_weight != 0.0 else "baseline"}_{args.dl_weight} \t P-Acc: {test_pred_acc:.2f}\t C-Acc: {test_constr_acc:.2f}\t Time [s]: {epoch_time:.1f} ({train_time:.1f})')

if __name__ == '__main__':
    main()