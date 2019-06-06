import tensorflow as tf
import numpy as np

# 8个词， 每个词向量32维度
embedding_inputs = tf.placeholder(tf.float32, shape=[None, 8, 32])
y_inputs = tf.placeholder(tf.int32, shape=[None,1])

# 卷积层
with tf.name_scope("cnn"):
    # CNN layer
    conv = tf.layers.conv1d(embedding_inputs, 16, 5, name='conv', padding='VALID')
    print(conv.shape)
    # global max pooling layer
    gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
    print(gmp.shape)

with tf.name_scope("score"):
    # 全连接层，后面接dropout以及relu激活
    fc = tf.layers.dense(gmp, 8, name='fc1')
    fc = tf.layers.dropout(fc, 0.9)
    fc = tf.nn.relu(fc)

    # 分类器
    logits = tf.layers.dense(fc, 2, name='fc2')
    # 预测类别
    y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)

with tf.name_scope("optimize"):
    # 损失函数，交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_inputs)
    loss = tf.reduce_mean(cross_entropy)
    # 优化器
    optim = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
