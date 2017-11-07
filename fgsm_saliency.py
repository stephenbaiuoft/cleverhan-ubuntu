"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging


from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.attacks import SaliencyMapMethod
from six.moves import xrange
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils_tf import model_train, model_eval, model_argmax
import timeit


import os

FLAGS = flags.FLAGS


# keep train_end same as source_samples
# since both models are donig the same for now
# epoch set to 2
def minist_fgsm_saliency(train_start=0, train_end=20, test_start=0,
                   test_end=5, nb_epochs=2, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64,
                   nb_classes=10,
                   source_samples=10,
                         ):
    """

    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    fout = open("saliency_map_summary.txt", "w+")
    ftime = open("timeer.txt", "w+")
    start = timeit.timeit()
    fout.write("starting running the overall script now\n\n\n")
    ftime.write("starting running the overall script now\n\n\n")


    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    # set_log_level(logging.DEBUG)

    # Create TF session
    sess = tf.Session()

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    # this way, all the 9 zeroes -> 0.1/9 because
    # the one-bit becomes 0.9
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))


    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    rng = np.random.RandomState([2017, 8, 30])


    ###########################################################################
    # Training the CNN model using TensorFlow: model --> base model
    ###########################################################################
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    # model for clean training
    model = make_basic_cnn(nb_filters=nb_filters)
    preds = model.get_probs(x)

    # model for fgsm method
    model_fgsm = make_basic_cnn(nb_filters=nb_filters)
    preds_fgsm = model.get_probs(x)

    # model for jocabian method
    model_jsma = make_basic_cnn(nb_filters=nb_filters)
    preds_jsma = model.get_probs(x)

    # model_of f and j
    model_f_j = make_basic_cnn(nb_filters=nb_filters)
    preds_f_j = model.get_probs(x)


    ###########################################################################
    # Clean Train ONLY!!
    ###########################################################################
    # model = make_basic_cnn(nb_filters=nb_filters)
    # preds = model.get_probs(x)
    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test
        # examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, args=eval_params)
        # report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Clean Train: Test accuracy on legitimate examples: %0.4f' % acc)

    ###########################################################################
    # MODEL Train!!!!!!!!!!!!
    ###########################################################################
    # training the basic model, using train_params


    model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                args=train_params, rng=rng)

    dur = timeit.timeit() - start
    start = timeit.timeit()
    ftime.write("\nBase Model Training Took: " + str(dur))

    # Calculate training error
    if testing:
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, x, y, preds, X_train, Y_train, args=eval_params)
            #report.train_clean_train_clean_eval = acc

    ###########################################################################
    # Generate FGSM Adversarial based on model, and
    # Attack Base Model Accuracy
    ###########################################################################
    fgsm_only = FastGradientMethod(model_fgsm, sess=sess)
    fgsm_params_y = {'eps': 0.3,
                     'y': y,
                   'clip_min': 0.,
                   'clip_max': 1.}

    adv_fgsm_only_x = fgsm_only.generate(x, **fgsm_params_y)
    preds_adv_fgsm_only = model_fgsm.get_probs(adv_fgsm_only_x)

    #jsma adversarial generating targeting fgsm trained model
    attack = SaliencyMapMethod(model_fgsm, back='tf', sess=sess)
    adv_x2 = attack.generate(x, **jsma_params)
    preds_adv_jsma_attack = model_fgsm.get_probs(adv_x2)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    # on 1250 examples for evaluation
    eval_par1 = {'batch_size': batch_size*10}
    acc = model_eval(sess, x, y, preds_adv_fgsm_only, X_test, Y_test, args=eval_par1)
    print('Test accuracy on FGSM Attack [Before Training]: %0.4f\n' % acc)
    fout.write("\nTest accuracy on FGSM Attack [Before Training]: " + str(acc))

    ###########################################################################
    # model_fgsm Training & Evaluation
    ###########################################################################

    def evaluate_fgsm():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_fgsm, X_test, Y_test,
                              args=eval_params)
        print('FGSM_Trained: Clean Data)Test accuracy: %0.4f' % accuracy)
        fout.write('\nFGSM_Trained Per Batch: Clean Data)Test accuracy: ' + str(accuracy))

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_adv_fgsm_only, X_test,
                              Y_test, args=eval_params)
        print('(FGSM_Trained: FGSM Data)Test accuracy: %0.4f' % accuracy)
        fout.write('\nFGSM_Trained Per Batch: FGSM Data)Test accuracy: ' + str(accuracy))

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_adv_jsma_attack, X_test,
                              Y_test, args=eval_params)
        print('(FGSM_Trained: JSMA Adversarial)Test accuracy: %0.4f' % accuracy)
        fout.write('\nFGSM_Trained Per Batch: JSMA Adversarial)Test accuracy: ' + str(accuracy))

    start = timeit.timeiit()
    # Perform and evaluate adversarial training FGSM
    model_train(sess, x, y, preds_fgsm, X_train, Y_train,
                predictions_adv=preds_adv_fgsm_only,evaluate=evaluate_fgsm,
                args=train_params, rng=rng)

    dur = timeit.timeit() - start
    ftime.write("\nFGSM Model Training Took: " + str(dur))

    #********************************SEPERATOR********************************#
    ###########################################################################
    # Generate JSMA Adversarial based on model, and
    # Attack Base Model Accuracy
    ###########################################################################
    # Instantiate a SaliencyMapMethod attack object
    jsma_only = SaliencyMapMethod(model_jsma, back='tf', sess=sess)
    # jsma_params, let target be randomly chosen
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    # create adv_saliency set tensor, using x_train data and jsma_params containing adv_y_target
    adv_jsma_only_x = jsma_only.generate(x, **jsma_params)
    # create adv preds tensor
    preds_adv_jsma_only = model.get_probs(adv_jsma_only_x)

    # Evaluate the accuracy of the MNIST model on JSMA adversarial examples
    # on 1250 examples for evaluation
    eval_par1 = {'batch_size': batch_size*10}
    acc = model_eval(sess, x, y, preds_adv_jsma_only, X_test, Y_test, args=eval_par1)
    print('Test accuracy on JSMA Attack [Before Training]: %0.4f\n' % acc)
    fout.write("\nTest accuracy on JSMA Attack [Before Training]: " + str(acc))

    #jsma adversarial generating targeting fgsm trained model
    attack = FastGradientMethod(model_jsma, sess=sess)
    adv_x2 = attack.generate(x, **fgsm_params)
    preds_adv_fgsm_attack = model_fgsm.get_probs(adv_x2)


    ###########################################################################
    # model_JSMA Training
    ###########################################################################
    def evaluate_jsma():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_jsma, X_test, Y_test,
                              args=eval_params)
        print('[JSMA_Trained: Clean Data]Test accuracy: %0.4f' % accuracy)
        fout.write("\n[JSMA_Trained: Clean Data]Test accuracy: " + str(accuracy))

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_adv_jsma_only, X_test,
                              Y_test, args=eval_params)
        print('[JSMA_Trained: JSMA Adversarial] Test accuracy: %0.4f' % accuracy)
        fout.write("\n[JSMA_Trained: JSMA Adversarial]Test accuracy: " + str(accuracy))

        # Accuracy of the adversarially trained model on FGSM examples
        accuracy = model_eval(sess, x, y, preds_adv_fgsm_attack, X_test,
                              Y_test, args=eval_params)
        print('[JSMA_Trained: FGSM Adversarial] Test accuracy: %0.4f' % accuracy)
        fout.write("\n[JSMA_Trained: FGSM Adversarial]Test accuracy: " + str(accuracy))

    start = timeit.timeit()
    # Perform and evaluate adversarial training FGSM
    model_train(sess, x, y, preds, X_train, Y_train,
                predictions_adv=preds_adv_jsma_only,
                evaluate=evaluate_jsma,
                args=train_params, rng=rng)

    dur = timeit.timeit() - start
    ftime.write("\nJSMA Model Training Took: " + str(dur))

    #***************************SEPERATOR*************************************#
    ###########################################################################
    # model_f_j based on model, and
    ###########################################################################
    # Initialize the Fast Gradient Sign Method (FGSM) attack MODEL_F_J
    fgsm = FastGradientMethod(model_f_j, sess=sess)
    fgsm_params_y = {'eps': 0.3,
                     'y': y,
                   'clip_min': 0.,
                   'clip_max': 1.}

    adv_fgsm_x = fgsm.generate(x, **fgsm_params_y)
    preds_adv_fgsm = model_f_j.get_probs(adv_fgsm_x)

    # Instantiate a SaliencyMapMethod attack object --> modify y_target for each test_data again
    jsma = SaliencyMapMethod(model_f_j, back='tf', sess=sess)
    # jsma_params, let target be randomly chosen
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    # create adv_saliency set tensor, using x_train data and jsma_params containing adv_y_target
    adv_jsma_x = jsma.generate(x, **jsma_params)
    # create adv preds tensor
    preds_adv_jsma = model_f_j.get_probs(adv_jsma_x)


    ###########################################################################
    # model_f_j training
    ###########################################################################
    start = timeit.timeit()
    # Perform and evaluate adversarial training FGSM
    model_train(sess, x, y, preds_f_j, X_train, Y_train,
                predictions_adv=preds_adv_fgsm,
                args=train_params, rng=rng)

    # Perform and evaluate adversarial training on JSMA
    model_train(sess, x, y, preds_f_j, X_train, Y_train,
                predictions_adv=preds_adv_jsma,
                args=train_params, rng=rng)

    dur = timeit.timeit() - start
    ftime.write("\nCombined Model Training Took: " + str(dur))

    # Calculate test error for combined model
    eval_par2 = {'batch_size': batch_size *10}
    acc_clean = model_eval(sess, x, y, preds_f_j, X_test,
                     Y_test, args=eval_par2)
    print("accuracy on clean test examples: ", acc_clean)
    fout.write("\n[Combined Model]accuracy on clean test examples: " + str(acc_clean))

    acc_fgsm = model_eval(sess, x, y, preds_adv_fgsm, X_test,
                     Y_test, args=eval_par2)
    print("accuracy on FGSM adversarial test examples: ", acc_fgsm)
    fout.write("\n[Combined Model]accuracy on FGSM adversarial test examples: " + str(acc_fgsm))



    acc_jsma = model_eval(sess, x, y, preds_adv_jsma, X_test,
                     Y_test, args=eval_par2)
    print("accuracy on JSMA adversarial test examples: ", acc_jsma)
    fout.write("\n[Combined Model]accuracy on JSMA adversarial test examples: " + str(acc_jsma))


    print("Overall Accuracy Combined: ", (acc_clean + acc_fgsm + acc_jsma)/3 )
    fout.write("\n\n\nOverall Accuracy Combined: " + str( (acc_clean + acc_fgsm + acc_jsma)/3))

    fout.close()
    ftime.close()

    return report




def main(argv=None):
    minist_fgsm_saliency(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   train_start=FLAGS.train_start,
                   train_end=FLAGS.train_end,
                   test_start=FLAGS.test_start,
                   test_end=FLAGS.test_end
                         )


if __name__ == '__main__':

    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 4, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_integer('train_start', 0, 'start of MNIST training samples')
    flags.DEFINE_integer('train_end', 60000, 'end of MNIST training samples')
    flags.DEFINE_integer('test_start', 0, 'start of MNIST test samples')
    flags.DEFINE_integer('test_end', 10000, 'end of MNIST test samples')

    # quick wrapper handles flag parsing and dispatches the main if defined
    tf.app.run()
    print("Tutorial Finished Running")