import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='dbbook2014',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[50, 100]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, e.g. 0,1,2...')

    parser.add_argument('--hidden_dim', type=int, default=128,
                        help="dimension of the semantic embedding to be retained")
    parser.add_argument('--ID_embs_init_type', type=str, default='normal',
                        help="ID embedding initialization type")
    parser.add_argument('--language_embs_scale', type=int, default=40,
                        help="language embedding scaling factor")
    parser.add_argument('--null_thres', type=str, default=None,
                        help="null threshold")
    parser.add_argument('--null_dim', type=int, default=64,
                        help="null dimension")
    parser.add_argument('--cover', type=bool, default=False,
                        help="whether to cover")
    parser.add_argument('--item_frequency_flag', type=bool, default=False,
                        help="use item frequency or not")
    parser.add_argument('--stand', type=bool, default=False,
                        help="whether to standardize")
    parser.add_argument('--init_from_pretrain', type=bool, default=False,
                        help="for original lgn")
    parser.add_argument('--plug_pretrain', type=float, default=0,
                        help="plus pretrain rate")

    # params4moe
    parser.add_argument('--loss_coef', type=float, default=0.001,
                        help="loss weight for moe loss")
    parser.add_argument('--moe_lr', type=float, default=0.001,
                        help="the learning rate of moe")
    parser.add_argument('--moe_epochs', type=int, default=10000)
    parser.add_argument('--num_experts', type=int, default=10,
                        help="number of experts")
    # parser.add_argument('--moe_hidden_size', type=int, default=512,
    #                     help="number of experts")
    parser.add_argument('--moe_hidden_size', type=int, nargs='+', default=[512],
                        help="hidden sizes for MLP layers, e.g., --moe_hidden_size 1024 200")
    parser.add_argument('--moe_output_size', type=int, default=64)
    parser.add_argument('--top_k', type=int, default=4,
                        help="number of selected experts")
    parser.add_argument('--router_type', type=str, default='joint',
                        help="available router_type: [joint, permod, disjoint]")

    return parser.parse_args()
