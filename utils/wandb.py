import omegaconf
from typing import Optional


def flatten_cfg(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            items.append((new_key, '-'.join(list(v))))
        elif isinstance(v, list):
            items.append((new_key, '-'.join(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def format_wandb_run_name(pattern: Optional[str], cfg: omegaconf.OmegaConf):
    if pattern is None:
        return None

    return pattern.format(**flatten_cfg(cfg))
