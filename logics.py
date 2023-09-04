import torch
from abc import ABC, abstractmethod

class Logic(ABC):
    @abstractmethod
    def LEQ(self, x, y):
        pass

    @abstractmethod
    def AND(self, x, y):
        pass

    @abstractmethod
    def OR(self, x, y):
        pass

    @abstractmethod
    def IMPL(self, x, y):
        pass

class BooleanLogic(Logic):
    def LEQ(self, x, y):
        return x <= y
    
    def AND(self, x, y):
        return torch.logical_and(x, y)
    
    def OR(self, x, y):
        return torch.logical_or(x, y)
    
    def IMPL(self, x, y):
        return torch.logical_or(torch.logical_not(x), y)

class DL2(Logic):
    def LEQ(self, x, y):
        return torch.clamp(x - y, min=0.0)
    
    def AND(self, x, y):
        return x + y
    
    def OR(self, x, y):
        return x * y

    def IMPL(self, x, y):
        return self.OR(1.0 - x, y)
    
class FuzzyLogic(Logic, ABC):
    # replaces zero values by a safe one to avoid division-by-zero and infinite gradients for sqrt
    def _safe_zero(self, x):
        eps = torch.finfo(x.dtype).eps
        return torch.where(x == 0.0, torch.full_like(x, eps), x)

    # range of fuzzy logic operators must be [0, 1]
    def _check(self, x):
        assert not torch.any((x < 0.0) | (x > 1.0))
        return x
    
    def LEQ(self, x, y):
        return self._check(self._LEQ(x, y))

    def AND(self, x, y):
        return self._check(self._AND(x, y))
    
    def OR(self, x, y):
        return self._check(self._OR(x, y))

    def IMPL(self, x, y):
        return self._check(self._IMPL(x, y))

    def _LEQ(self, x, y):
        eps = 0.05
        return 1.0 - torch.clamp(x-y, min=0.0) / (torch.abs(x) + torch.abs(y) + eps)
    
    @abstractmethod
    def _AND(self, x, y):
        pass

    @abstractmethod
    def _OR(self, x, y):
        pass

    @abstractmethod
    def _IMPL(self, x, y):
        pass
    
class GödelFuzzyLogic(FuzzyLogic):
    def _AND(self, x, y):
        return torch.minimum(x, y)

    def _OR(self, x, y):
        return torch.maximum(x, y)

    def _IMPL(self, x, y):
        return torch.where(x < y, 1.0, y)
    
class KleeneDienesFuzzyLogic(GödelFuzzyLogic):
    def _IMPL(self, x, y):
        return torch.maximum(1.0 - x, y)
    
class LukasiewiczFuzzyLogic(FuzzyLogic):
    def _AND(self, x, y):
        return torch.maximum(torch.zeros_like(x), x + y - 1.0)
    
    def _OR(self, x, y):
        return torch.minimum(torch.ones_like(x), x + y)

    def _IMPL(self, x, y):
        return torch.minimum(1.0 - x + y, torch.ones_like(x))
    
class ReichenbachFuzzyLogic(FuzzyLogic):
    def _AND(self, x, y):
        return x * y
    
    def _OR(self, x, y):
        return x + y - x * y

    def _IMPL(self, x, y):
        return 1.0 - x + x * y
    
class GoguenFuzzyLogic(ReichenbachFuzzyLogic):
    def _IMPL(self, x, y):
        return torch.where(torch.logical_or(x <= y, x == 0.0), torch.tensor(1.0, device=x.device), y/self._safe_zero(x))
    
class ReichenbachSigmoidalFuzzyLogic(ReichenbachFuzzyLogic):
    def __init__(self, s = 2):
        self.s = s

    def _IMPL(self, x, y):
        exp = torch.exp(torch.tensor([self.s / 2], device=x.device))

        numerator = (1.0 + exp) * torch.sigmoid(self.s * super()._IMPL(x, y) - self.s/2) - 1.0
        denominator = exp - 1.0

        I_s = torch.clamp(numerator / self._safe_zero(denominator), max=1.0)

        return self._check(I_s)
    
class ReichenbachBijectionFuzzyLogic(ReichenbachFuzzyLogic):
    def __init__(self, phi = torch.square, phi_inverse = None):
        self.phi = phi
        self.phi_inverse = phi_inverse if phi_inverse is not None else lambda x: torch.sqrt(self._safe_zero(x))

    def _IMPL(self, x, y):
        return self._check(self.phi_inverse(super()._IMPL(self.phi(x), self.phi(y))))
    
class YagerFuzzyLogic(FuzzyLogic):
    def __init__(self, p = 2):
        self.p = p

    def _AND(self, x, y):
        return torch.clamp(1 - torch.pow(self._safe_zero(torch.pow(1.0 - x, self.p) + torch.pow(1.0 - y, self.p)), 1.0/self.p), min=0.0)
    
    def _OR(self, x, y):
        return torch.clamp(torch.pow(torch.pow(x, self.p) + torch.pow(y, self.p), 1.0/self.p), max=1.0)

    def _IMPL(self, x, y):
        return torch.where(torch.logical_and(x == 0.0, y == 0.0), torch.ones_like(x), torch.pow(y, x))