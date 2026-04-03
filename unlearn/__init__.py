from .baseline import Baseline
from .retrain import Retrain
from .finetune import Finetune
from .gradient_ascent import GradAscent
from .bad_teacher import BadTeacher
from .random_label import RandomLabel
from .scrub import SCRUB
from .salun import SalUn
from .sfron import SFRon
from .unlearn_method import UnlearnMethod

def create_unlearn_method(unlearn_name) -> UnlearnMethod:
    return eval(unlearn_name)