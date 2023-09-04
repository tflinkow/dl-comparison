from functools import reduce
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from labels import fashion_mnist_labels, cifar10_labels, gtsrb_labels
from logics import BooleanLogic

class Constraint(ABC):
    def set_model_output(self, output):
        self.x_out = output
        self.probabilities = F.softmax(self.x_out, dim=1)
        self.boolean_logic = BooleanLogic()

    @abstractmethod
    def constraint(self, logic):
        pass

    # usage:
    # loss, sat = eval()
    # where sat returns whether the constraint is satisfied or not
    def eval(self, logic):
        return self.constraint(logic), self.constraint(self.boolean_logic)

class ClassSimilarityConstraint(Constraint):
    def __init__(self, eps, indices):
        self.eps = eps
        self.indices = indices

    def constraint(self, logic):
        # NOTE: ensure that logic._AND is the same for all fuzzy logics in order to only compare IMPL
        return reduce(logic.AND, [logic.IMPL(logic.LEQ(self.eps, self.probabilities[:, i[0]]), logic.LEQ(self.probabilities[:, i[2]], self.probabilities[:, i[1]])) for i in self.indices])

class GroupConstraint(Constraint):
    def __init__(self, eps, group_indices):
        self.eps = eps
        self.group_indices = group_indices

    def constraint(self, logic):
        groups = [torch.sum(self.probabilities[:, indices], dim=1) for indices in self.group_indices]

        # NOTE: ensure that logic._OR is the same for all fuzzy logics in order to only compare AND
        return reduce(logic.AND, [logic.OR(logic.LEQ(g, self.eps), logic.LEQ(1.0 - self.eps, g)) for g in groups])
        
def FashionMnistConstraint(eps):
    groups = [
        # [a, b, c]
        # if target=a then out_b >= out_c
        ['T-shirt/top', 'Shirt', 'Ankle boot'],
        ['Trouser', 'Dress', 'Bag'],
        ['Pullover', 'Shirt', 'Sandal'],
        ['Dress', 'Coat', 'Bag'],
        ['Coat', 'Pullover', 'Shirt'],
        ['Sandal', 'Sneaker', 'Dress'],
        ['Shirt', 'Pullover', 'Sneaker'],
        ['Sneaker', 'Sandal', 'Trouser'],
        ['Bag', 'Sandal', 'Dress'],
        ['Ankle boot', 'Sneaker', 'T-shirt/top'],
    ]

    return ClassSimilarityConstraint(eps, [[fashion_mnist_labels[e] for e in g] for g in groups])

def Cifar10Constraint(eps):
    groups = [
        ['airplane', 'ship', 'dog'],
        ['automobile', 'truck', 'cat'],
        ['bird', 'airplane', 'dog'],
        ['cat', 'dog', 'frog'],
        ['deer', 'horse', 'truck'],
        ['dog', 'cat', 'bird'],
        ['frog', 'ship', 'truck'],
        ['horse', 'deer', 'airplane'],
        ['ship', 'airplane', 'deer'],
        ['truck', 'automobile', 'airplane']
    ]

    return ClassSimilarityConstraint(eps, [[cifar10_labels[e] for e in g] for g in groups])

def GtsrbConstraint(eps):
    groups = [
        ['limit 20km/h', 'limit 30km/h', 'limit 50km/h', 'limit 60km/h', 'limit 70km/h', 'limit 80km/h', 'end of limit 80km/h', 'limit 100km/h', 'limit 120km/h', ],
        ['turn right ahead', 'turn left ahead', 'ahead only', 'go straight or right', 'go straight or left', 'keep right', 'keep left', 'roundabout'],
        ['no passing', 'no passing for trucks', 'no way', 'no way one-way', 'end of no passing', 'end of no passing for trucks'],
        ['caution general', 'caution curve left', 'caution curve right', 'caution curvy', 'caution bumps', 'caution slippery', 'caution narrow road', 'road work', 'pedestrians', 'children crossing', 'wild animals crossing']
    ]

    return GroupConstraint(eps, [[gtsrb_labels[e] for e in g] for g in groups])