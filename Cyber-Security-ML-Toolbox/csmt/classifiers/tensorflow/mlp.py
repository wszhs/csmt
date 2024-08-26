
from csmt.classifiers.abstract_model import AbstractModel

class MLPTensor(AbstractModel):
    def __init__(self,input_size,output_size):
        import tensorflow as tf
        # tf.compat.v1.disable_eager_execution()
        from csmt.estimators.classification.tensorflow import TensorFlowClassifier,TensorFlowV2Classifier
        input_ph = tf.placeholder(tf.float32, shape=[None, input_size])
        labels_ph = tf.placeholder(tf.int32, shape=[None, output_size])

        x = tf.layers.dense(input_ph,64,activation=tf.nn.relu)
        x = tf.layers.dense(x, 32, activation=tf.nn.relu)
        logits = tf.layers.dense(x, output_size)

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