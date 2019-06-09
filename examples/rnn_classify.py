#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64  # 词向量维度
    seq_length = 100  # 序列长度
    num_classes = 10  # 类别数
    vocab_size = 5000  # 词汇表达小

    num_layers = 2  # 隐藏层层数
    hidden_dim = 128  # 隐藏层神经元
    rnn = 'gru'  # lstm 或 gru

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextRNN(object):
    """文本分类，RNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)

        # 为每一个rnn核后面加一个dropout层
        def dropout():
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('rnn'):
            cells = [dropout() for i in range(self.config.num_layers)]

            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                 inputs=embedding_inputs,
                                                 dtype=tf.float32)
            last = _outputs[:,-1,:]
            # print(_outputs)
            # print(states)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1', activation='relu')
            fc = tf.layers.dropout(fc, self.keep_prob)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2', activation='softmax')
            self.y_pred_cls = tf.argmax(self.logits, 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    config = TRNNConfig()
    rnn = TextRNN(config)
