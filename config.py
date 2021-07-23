import argparse

parser = argparse.ArgumentParser(description='PyTorch Example for all')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='whether to use cuda to accerlate')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--num_class', type=int, default=4)
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--mlp_dim', type=int, default=3072)
parser.add_argument('--num_heads', type=int, default=12)
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--attn_dropout_rate', type=float, default=0.0)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-4)


