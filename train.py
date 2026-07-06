
from training.trainer import train
from utils.util_funcs import parse_args

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
    