
from csmt.classifiers.abstract_model import AbstractModel

class CNNTensor(AbstractModel):
    def __init__(self,input_size,output_size):
        # tf.compat.v1.disable_eager_execution()
        from csmt.estimators.classification.tensorflow import TensorFlowClassifier
        import tensorflow as tf
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        labels_ph = tf.placeholder(tf.int32, shape=[None, 10])

        x = tf.layers.conv2d(input_ph, filters=4, kernel_size=5, activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, filters=10, kernel_size=5, activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 100, activation=tf.nn.relu)
        logits = tf.layers.dense(x, 10)

        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.classifier = TensorFlowClassifier(
            input_ph=input_ph,
            output=logits,
            labels_ph=labels_ph,
            train=train,
            loss=loss,
            learning=None,
            sess=sess
        )