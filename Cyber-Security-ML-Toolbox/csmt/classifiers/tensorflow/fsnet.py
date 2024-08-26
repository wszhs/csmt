
from csmt.classifiers.abstract_model import AbstractModel

class FsnetTensor(AbstractModel):
    def __init__(self,input_size,output_size):
        import tensorflow as tf
        # tf.compat.v1.disable_eager_execution()
        from csmt.estimators.classification.tensorflow import TensorFlowClassifier
        from .fsnet_base import Fs_net
        input_ph = tf.placeholder(tf.float32, [None, input_size, 1])
        labels_ph = tf.placeholder(tf.int32, [None])

        fs_net = Fs_net(input_ph, labels_ph,output_size)
        loss, logits = fs_net.build_loss()  

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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