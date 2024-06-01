import os
from utils.logging import logger

from omegaconf import OmegaConf

# Retrieve the configs path
conf_path = os.path.join(os.path.dirname(__file__), '../configs')

# Read the cli args
cli_args = OmegaConf.from_cli()

# read a specific config file
args = OmegaConf.create()
if 'config' in cli_args and cli_args.config:
    base_conf_path = cli_args.config
else:
    base_conf_path = "debug.yaml"

logger.info(f"Loading configuration from {base_conf_path}!")

while base_conf_path is not None:
    base_conf = OmegaConf.load(os.path.join(conf_path, base_conf_path))
    base_conf_path = base_conf.extends
    args = OmegaConf.merge(base_conf, args)

# Merge cli args into config ones
args = OmegaConf.merge(args, cli_args)
