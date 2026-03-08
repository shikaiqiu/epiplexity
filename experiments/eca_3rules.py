
from soph.train import train
from experiments.ca_shared import patch_predict_half, run
from soph.utils.config_generator import dispatch_multigpu, grid_iter
import copy

if __name__ == "__main__":
    debug = False

    subset = [15, 30, 54]

    cfg = copy.deepcopy(train.__kwdefaults__)
    width = 64
    cfg.update({
        'arch': 'gpt',
        'L': [1, 2, 4, 6, 9],
        'D': [16, 32, 64, 128, 256, 512],
        'd_head': 64,
        'T': 10000,
        'B': 384,
        'A': 4,
        'warmup': 100,
        'tag': 'arxiv_3rules',
        'compile': False,
        'wandb_log': not debug,
        'wandb_project': 'requential',
        'log_geometric': True,
        'requential': True,
        'student_speed': 1,
        'ema_steps': 50,
        'data_cfg': {'steps': [48],'rule': subset, 'width': width},
        'apply_patch': patch_predict_half,
    })

    if debug:
        nextcfg = list(grid_iter(cfg))[0]
        run(nextcfg)
    else:
        dispatch_multigpu(run,cfg,ordered=True)
