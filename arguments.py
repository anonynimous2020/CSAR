import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')

    parser.add_argument('--approach', default='', type=str, required=True,
                        choices=['BERT',
                                 'PALS',
                                 'UCL',
                                 'CSAR',
                                 'FreezeBert',
                                 'LSTM',
                                 'CNN_Text',
                                 'random', 
                                 'sgd', 
                                 'sgd-frozen', 
                                 'sgd_with_log', 
                                 'sgd_L2_with_log', 
                                 'lwf','lwf_with_log', 
                                 'lfl',
                                 'ewc', 
                                 'si', 
                                 'rwalk', 
                                 'mas', 
                                 'ucl', 
                                 'ucl_ablation', 
                                 'baye_fisher',
                                 'baye_hat', 
                                 'imm-mean', 
                                 'progressive', 
                                 'pathnet',
                                 'imm-mode', 
                                 'sgd-restart', 
                                 'joint', 
                                 'hat', 
                                 'hat-test'], 
                        help='(default=%(default)s)')
    parser.add_argument('--logname',default='',type=str, required=True, help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--ratio', default='0.5', type=float, help='(default=%(default)f)')
    parser.add_argument('--alpha', default=1, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default=1, type=float, help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.03, type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--tasks_sequence', type = int, default=1)
    args=parser.parse_args()
    return args