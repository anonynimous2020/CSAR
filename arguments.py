import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')

    parser.add_argument('--approach', default='', type=str, required=True,
                        choices=['BERT',
                                 'PALS',
                                 'UCL',
                                 'CSUR',
                                 'FreezeBert',
                                 'LSTM',
                                 'CNN_Text'
                                 ], 
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