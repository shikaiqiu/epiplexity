
from soph.train import train
from experiments.ca_shared import patch_predict_half, run
from soph.utils.config_generator import dispatch_multigpu, grid_iter
import copy

if __name__ == "__main__":
    debug = False

    rules = [0, 32, 160, 232, # class 1: convergent
            4, 15, 108, 218, # class 2: periodic
            22, 30, 126, 150, # class 3: chaotic
            41, 54, 106, 110] # class 4: complex

    subset = [0, 32, 4, 15, 22, 30, 41, 54, 106, 110]

    cfg = copy.deepcopy(train.__kwdefaults__)
    width = 64
    cfg.update({
        'arch': 'gpt',
        'L': [1, 2, 3],
        'D': [16, 32, 64, 128],
        'd_head': 32,
        'T': 10000,
        'B': 96*4 * width // 64,
        'A': 1,
        'warmup': 100,
        'tag': 'eca_rules',
        'compile': False,
        'wandb_log': not debug,
        'wandb_project': 'requential',
        'log_geometric': True,
        'requential': False,
        'student_speed': 1,
        'ema_steps': 50,
        'data_cfg': {'steps': [128],'rule': subset, 'width': width},
        'apply_patch': patch_predict_half,
    })

    if debug:
        nextcfg = list(grid_iter(cfg))[0]
        run(nextcfg)
    else:
        dispatch_multigpu(run,cfg,ordered=True)
