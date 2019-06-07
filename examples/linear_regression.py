import tensorflow as tf
import numpy as np

# 产生数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
y_data = x_data * 2 + 3 + np.random.normal(0, 0.02)

# 输入数据的placeholder
x_input = tf.placeholder(tf.float32,[None, 1], name="x_input")
y_input = tf.placeholder(tf.float32,[None, 1], name="y_input")
# 定义需要训练的参数
w = tf.Variable(tf.random_normal([1]), name="w")
b = tf.Variable(tf.random_normal([1]), name="b")
# 定义模型
y = x_input * w + b
# 定义损失函数
loss = tf.reduce_mean(tf.pow(y - y_input, 2))
# 优化损失函数
min = tf.train.AdamOptimizer(0.1).minimize(loss)

# 初始化所有参数
init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter('graph',sess.graph)
tf.summary.scalar("loss",loss)
merge = tf.summary.merge_all()
sess.run(init)
# 输出初始化化好的参数
print("W=", sess.run(w), "b=", sess.run(b))
for step in range(10):
    summary, _ = sess.run([merge, min], feed_dict={x_input: x_data, y_input: y_data})
    writer.add_summary(summary,step)
writer.close()
# 输出训练好的参数
print("W=", sess.run(w), "b=", sess.run(b))