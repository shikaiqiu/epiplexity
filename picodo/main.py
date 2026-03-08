import hydra
import train
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='configs', config_name='default')
def main(c: DictConfig):
    train.train_and_evaluate(c)


if __name__ == '__main__':
    main()
