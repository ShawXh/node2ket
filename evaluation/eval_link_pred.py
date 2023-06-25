from EmbLoader import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str)
    parser.add_argument('--conv-emb', type=str)
    parser.add_argument('--tu-emb', type=str)
    parser.add_argument('--tu-config', type=str, default=None, help="tu config path")
    parser.add_argument('-c', '--eval-conv-emb', default=False, action="store_true", help="evaluate conventional embedding")
    parser.add_argument('-t', '--eval-tu-emb', default=False, action="store_true", help="evaluate tu embedding")
    return parser.parse_args()

args = parse_args()

network_file = args.net
masked_link_file = args.net + "-edges"
noise_link_file = args.net[:-6] + "noise-edges"

if args.eval_conv_emb:
    convemb_file = args.conv_emb
    tuemb_file = None
    tuconfig_file = None
elif args.eval_tu_emb:
    convemb_file = None
    tuemb_file = args.tu_emb
    tuconfig_file = args.tu_config


network = Network(network_file)
network.task_link_pred(
    masked_link_file=masked_link_file, 
    noise_link_file=noise_link_file, 
    conv_emb_file=convemb_file,
    tu_emb_file=tuemb_file,
    tu_config_file=tuconfig_file)
