from .attack_framework import Attack_Framework
from .Apollo import Apollo, Apollo_Offline
from .ULiRA import ULiRA
from .UMIA import UMIA

def get_attack(name, **kwargs) -> Attack_Framework:
    return eval(name)(**kwargs)