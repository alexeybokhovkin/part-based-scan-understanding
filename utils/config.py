import yaml
import argparse


def _load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)


def load_config(args):
    parser = argparse.ArgumentParser(description='Residual3DUnet training/testing scripts')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parsed_args = parser.parse_args(args)
    config = _load_config_yaml(parsed_args.config)
    if 'resume_from_checkpoint' not in config:
        config['resume_from_checkpoint'] = None
    return argparse.Namespace(**config)
