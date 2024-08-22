import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file_path', type=str, default='./yaml_config/base.yaml', help='Name of the experiment configuration path')

    args = parser.parse_args()

    return args

