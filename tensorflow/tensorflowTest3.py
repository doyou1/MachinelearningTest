import tensorflow as tf

#xData : 노동시간
xData = [1,2,3,4,5,6,7]
#yData : 수익
yData = [25000,55000,75000,110000,128000,155000,180000]
W = tf.Variable(tf.random_uniform([1],-100,100))

b = tf.Variable(tf.random_uniform([1],-100,100))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

H = W * X + b
cost = tf.reduce_mean(tf.square(H - Y))

#하강 점프 길이 정함
a = tf.Variable(0.01)
#경사하강 라이브러리
optimizer = tf.train.GradientDescentOptimizer(a)

train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(5001):
    #실제 학습
    sess.run(train, feed_dict={X: xData, Y: yData})
    if i % 500 == 0:
        #학습의 횟수,, 학습한 데이터
        print(i, sess.run(cost, feed_dict={X: xData, Y: yData}), sess.run(W), sess.run(b))

#session에 실제 데이터를 넣어 나온 답 15시간 일하면 얼마를 버나?
print (sess.run(H, feed_dict={X: [15]}))