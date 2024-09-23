import argparse
import ipdb
class MainServer():
    def __init__(args, self):
        self.modelserver = dict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat', type=str, default='')
    parser.add_argument('--a2t', type=str, default='')
    parser.add_argument('--tfg', type=str, default='')
    parser.add_argument('--tts', type=str, default='')
    args = parser.parse_args()
    
