from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans_tutorials.tutorial_models import make_basic_cnn
import timeit

FLAGS = flags.FLAGS


def mnist_tutorial_jsma(train_start=0, train_end=60000, test_start=0,
                        test_end=10000, viz_enabled=True, nb_epochs=6,
                        batch_size=128, nb_classes=10, source_samples=10,
                        learning_rate=0.001):
    """
    MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """
    f_out = open("jsma_test.log", "w");
    start = timeit.timeit()
    f_out.write(" JSMA Training Started:\n")


    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2017, 8, 30])
    model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                rng=rng)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')


    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}



    # create adv_saliency set tensor, using x_train data and jsma_params containing adv_y_target
    adv_jsma_only_x = jsma.generate(x, **jsma_params)
    # create adv preds tensor
    preds_jsma = model.get_probs(adv_jsma_only_x)

    def evaluate_jsma():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                              args=eval_params)
        print('[JSMA_Trained: Clean Data]Test accuracy: %0.4f' % accuracy)

        dur = timeit.timeit() - per_start
        f_out.write("\ntime it took is: ", dur,
                    "\n[JSMA_Trained: Clean Data]Test accuracy: " + str(accuracy))

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_jsma, X_test,
                              Y_test, args=eval_params)
        print('[JSMA_Trained: JSMA Adversarial] Test accuracy: %0.4f' % accuracy)

        dur = timeit.timeit() - per_start
        f_out.write("\nJSMA time it took is: ", dur,
                    "\n[JSMA_Trained: JSMA Adversarial]Test accuracy: " + str(accuracy))

    per_start = timeit.timeit()
    # Perform and evaluate adversarial training FGSM
    model_train(sess, x, y, preds, X_train, Y_train,
                predictions_adv=preds_jsma,
                evaluate=evaluate_jsma,
                args=train_params, rng=rng)

    dur = timeit.timeit() - start
    print("\nScript Finished Running\nTotal Duration was: ", dur)
    f_out.write("total training took: ", dur)
    f_out.close()

    # Close TF session
    sess.close()

    return report


def main(argv=None):
    mnist_tutorial_jsma(viz_enabled=FLAGS.viz_enabled,
                        nb_epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        nb_classes=FLAGS.nb_classes,
                        source_samples=FLAGS.source_samples,
                        learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    tf.app.run()
