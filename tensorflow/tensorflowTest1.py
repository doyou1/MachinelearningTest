import tensorflow as tf

a = tf.constant(17)
b = tf.constant(5)

#덧셈
c = tf.add(a,5)
sess = tf.Session()
print(sess.run(c))

#뺄셈
c = tf.subtract(a,5)
print(sess.run(c))

#곱셈
c = tf.multiply(a,5)
print(sess.run(c))

#나눗셈
c = tf.truediv(a,5)
print(sess.run(c))

#나머지
c = tf.mod(a,5)
print(sess.run(c))

#절대값
c = tf.abs(-a)
print(sess.run(c))

a = tf.constant(-17.5)
b = tf.constant(5.0)
# 음수로 리턴
c = tf.negative(a)
print(sess.run(c))

# 부호 리턴, 양수: 1,음수:-1
c = tf.sign(a)
print(sess.run(c))

# 제곱\
c = tf.square(a)
print(sess.run(c))

# 제곱근
c = tf.sqrt(a)
print(sess.run(c))

# 거듭제곱
c = tf.pow(b,2)
print(sess.run(c))

# 최대값
c = tf.maximum(a,b)
print(sess.run(c))

# 최솟값
c = tf.minimum(a,b)
print(sess.run(c))

# 지수 값
c = tf.exp(b)
print(sess.run(c))

# 로그 값
c = tf.log(b)
print(sess.run(c))

# 사인 함수
c = tf.sin(b)
print(sess.run(b))

# 코사인 함수
c = tf.cos(b)
print(sess.run(c))