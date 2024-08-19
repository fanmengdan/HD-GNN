# -*- coding: utf-8 -*-
import argparse
import os
import scipy.misc
import numpy as np
from model_2 import graph2graph
import tensorflow.compat.v1 as tf
FLAGS = tf.app.flags.FLAGS

def main(_):
    steps = [2, 3, 5]
    entity_nodes = [200, 250, 250]
    hunk_nodes = [74, 114, 150]
    entity_edges = [39800, 62250, 62250]
    hunk_edges = [5402, 12882, 22350]

    # 使用zip函数将五个列表组合起来
    zipped = zip(steps, entity_nodes, hunk_nodes, entity_edges, hunk_edges)

    # 遍历zip对象
    for step, entity_node, hunk_node, entity_edge, hunk_edge in zipped:
        print(step, entity_node, hunk_node, entity_edge, hunk_edge)

        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epoch', type=int, default=50, help='number of training epochs')  # 50
        parser.add_argument('--Ds', type=int, default=1, help='The State Dimention')
        parser.add_argument('--Ds_inter', type=int, default=1, help='The State Dimention of inter state')
        parser.add_argument('--Dr', type=int, default=2, help='The Relationship Dimension')
        parser.add_argument('--Dr_inter', type=int, default=2, help='The Relationship Dimension of inter state')
        # parser.add_argument('--Dx', type=int, default=3,help='The External Effect Dimension')
        parser.add_argument('--De_e', type=int, default=20, help='The Effect Dimension on entity')
        parser.add_argument('--De_er', type=int, default=20, help='The Effect Dimension on entity Relations')
        parser.add_argument('--Mini_batch', type=int, default=50, help='The training mini_batch')  # ！！！
        # parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint40/glide',
        #                     help='models are saved here')
        parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint40/',
                            help='models are saved here')

        parser.add_argument('--Ne', type=int, default=entity_node, help='The Number of entities')  # 200, 250, 250
        parser.add_argument('--Nc', type=int, default=hunk_node, help='The Number of code changes')  # 74, 114, 150
        parser.add_argument('--Ner', type=int, default=entity_edge,
                            help='The Number of entity Relations')  # 39800, 62250, 62250
        parser.add_argument('--Ncr', type=int, default=hunk_edge,
                            help='The Number of code change Relations')  # 5402, 12882, 22350
        parser.add_argument('--Step', type=int, default=step, help='the number of commits/groups')  # 2, 3, 5
        parser.add_argument('--Repo', type=str, default='glide', help='the name of repository')

        parser.add_argument('--Type', dest='Type', default='train', help='train or test')
        args = parser.parse_args()

        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        tf.reset_default_graph()
        with tf.Session() as sess: #sess就像一个指针，处理的地方被激活
            # 搭建 model
            model = graph2graph(sess,
                                Ds=args.Ds,
                                Ne=args.Ne, Nc=args.Nc,
                                Ner=args.Ner, Ncr=args.Ncr,
                                Dr=args.Dr,
                                De_e=args.De_e,De_er=args.De_er,
                                Mini_batch=args.Mini_batch,
                                checkpoint_dir=args.checkpoint_dir,
                                epoch=args.epoch,
                                Ds_inter=args.Ds_inter,Dr_inter=args.Dr_inter,
                                Step=args.Step,
                                Repo=args.Repo
                                )
            # sess.run model 里面的变量
            if args.Type == 'train':
               model.train(args) # input 入口
            if args.Type == 'test':
               model.test(args)

if __name__ == '__main__':
      tf.app.run()