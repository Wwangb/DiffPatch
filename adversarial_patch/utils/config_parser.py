"""
Create argparse options for config files
"""
import json
import argparse
from easydict import EasyDict as edict


def load_config_object(cfg_path: str) -> edict:
    """
    Loads a config json and returns a edict object
    """
    with open(cfg_path, 'r', encoding="utf-8") as json_file:
        cfg_dict = json.load(json_file)
    
    return edict(cfg_dict)


def get_argparser(desc="Config file load for training adv patches") -> argparse.ArgumentParser:
    """
    Get parser with the default config argument
    """
    parser = argparse.ArgumentParser(
        description=desc)
    parser.add_argument('--cfg', '--config', type=str, dest="config",
                        default="/remote-home/wangzhixiang/experiments/blended-latent-diffusion/yolov5_adversarial/adv_patch_gen/configs/base.json",
                        help='Path to JSON config file to use for adv patch generation (default: %(default)s)')
    return parser

