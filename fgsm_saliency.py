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

import os

FLAGS = flags.FLAGS


# keep train_end same as source_samples
# since both models are donig the same for now
# epoch set to 2
def minist_fgsm_saliency(train_start=0, train_end=10, test_start=0,
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

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

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

    # placeholder for y_target --> for saliency tensor
    y_target = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])


    ###########################################################################
    # Training the CNN model using TensorFlow: model --> base model
    ###########################################################################
    model = make_basic_cnn(nb_filters=nb_filters)
    preds = model.get_probs(x)


    if clean_train:
        # omg -> creates a cnn model
        # model = make_basic_cnn(nb_filters=nb_filters)
        # preds = model.get_probs(x)
        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)

        ###########################################################################
        # MODEL Train!!!!!!!!!!!!
        ###########################################################################
        # training the basic model, using train_params
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc


        ###########################################################################
        # Generate FGSM Adversarial based on model, and
        # Compute Base Model Accuracy
        ###########################################################################

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)

        # todo: follow the paper and run Cleverhans Output?
        fgsm_params_y = {'eps': 0.3,
                         'y': y,
                       'clip_min': 0.,
                       'clip_max': 1.}

        #adv_x = fgsm.generate(x, **fgsm_params)
        adv_x = fgsm.generate(x, **fgsm_params_y)
        preds_adv = model.get_probs(adv_x)
        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on FGSM adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc

        ###########################################################################
        # Generate Saliency Map Adversarial Example and
        # Compute base model accuracy (only 10)
        ###########################################################################
        print("Saliency Map Attack On The Base Model")
        print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes - 1) +
              ' adversarial examples')

        # Instantiate a SaliencyMapMethod attack object --> modify y_target for each test_data again
        jsma = SaliencyMapMethod(model, back='tf', sess=sess)
        jsma_params = {'theta': 1., 'gamma': 0.1,
                       'clip_min': 0., 'clip_max': 1.,
                       'y_target': None}

        # Keep track of success (adversarial example classified in target)
        # Need this info to compute the success rate
        results = np.zeros((nb_classes, source_samples), dtype='i')

        # each sample will get 9 adversarial samples

        # adv_x_set: place_holder for all the x variations
        # correct_y_set: correct_y_output used for training

        adv_x_set = None
        adv_y_target = None

        # we need multi x_train_saliency / y_train_saliency
        #
        x_train_saliency = None
        y_train_saliency = None

        for sample_ind in xrange(0, source_samples):
            print('--------------------------------------')
            print('Saliency Attacking input %i/%i' % (sample_ind + 1, source_samples))
            sample = X_train[sample_ind:(sample_ind + 1)]
            y_sample = Y_train[sample_ind:(sample_ind + 1)]

            current_class = int(np.argmax(Y_train[sample_ind]))
            target_classes = other_classes(nb_classes, current_class)

            # Loop over all target classes
            for target in target_classes:
                print('Generating adv. example for target class %i' % target)

                # Create x_train_saliency, corresponding to y_train_saliency
                if x_train_saliency is not None:
                    x_train_saliency = np.concatenate((x_train_saliency, sample), axis=0)
                    y_train_saliency = np.concatenate((y_train_saliency, y_sample), axis=0)
                else:
                    x_train_saliency = sample
                    y_train_saliency = y_sample
                    print("sample shape: ", x_train_saliency.shape)
                    print("y_sample shape: ", y_train_saliency.shape)


                # This call runs the Jacobian-based saliency map approach
                one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
                one_hot_target[0, target] = 1
                jsma_params['y_target'] = one_hot_target

                adv_x_np = jsma.generate_np(sample, **jsma_params)

                # Add to adv_x_set, correct_y_set
                if adv_x_set is not None:
                    adv_y_target = np.concatenate((adv_y_target, one_hot_target), axis=0)
                    adv_x_set = np.concatenate((adv_x_np, adv_x_set), axis=0)
                else:
                    adv_y_target = one_hot_target
                    adv_x_set = adv_x_np
                    print("adv_y_target shape(one-hot-encoding): ", adv_y_target.shape)
                    print("adv_x_set(np) shape: ", adv_x_np.shape)

                # Check if success was achieved
                res = int(model_argmax(sess, x, preds, adv_x_np) == target)

                # Update the arrays for later analysis
                results[target, sample_ind] = res

        print('--------------------------------------')
        # Compute the number of adversarial examples that were successfully found
        nb_targets_tried = ((nb_classes - 1) * source_samples)
        succ_rate = float(np.sum(results)) / nb_targets_tried
        print('Avg. rate of successful Saliency adv. examples {0:.4f}'.format(succ_rate))
        report.clean_train_adv_eval = 1. - succ_rate

        # here we have successfully stacked up x_adversarial_set, y_correct_set
        # these can be used to provide training to our model now
        print("\n\n\n*****************************")
        print("Checking x_adv_set shape: ", adv_x_set.shape)
        print("Checking correct_y_set shape: ", adv_y_target.shape)

        print("x_training_saliency shape:", x_train_saliency.shape)
        print("y_training_saliency shape:", y_train_saliency.shape)

        # now construct model 3, define output -> input relationship tensor
        model_3 = make_basic_cnn(nb_filters=nb_filters)
        # define the x, the placeholder input - > preds_3 output
        preds_3 = model_3(x)

        # jsma3 = SaliencyMapMethod(model_3, sess=sess)
        #
        # jsma_params = {'theta': 1., 'gamma': 0.1,
        #                'clip_min': 0., 'clip_max': 1.,
        #                'y_target': y_target}
        #
        # # create adv_saliency set tensor, using x_train data and jsma_params containing adv_y_target
        # adv_jsma = jsma3.generate(x, jsma_params)
        # # create adv preds tensor
        # preds_jsma_adv = model_3(adv_jsma)

        # define saliency training model accuracy
        def evaluate_saliency():
            # Accuracy of adversarially trained model on legitimate test inputs
            eval_params = {'batch_size': batch_size}
            accuracy = model_eval(sess, x, y, preds_3, x_train_saliency, y_train_saliency,
                                  args=eval_params)
            print('Test accuracy on legitimate examples: %0.4f' % accuracy)
            report.adv_train_clean_eval = accuracy

        ###########################################################################
        # MODEL Train for Saliency Map
        ###########################################################################
        # Perform and evaluate adversarial training with FSGM MODEL!!!
        # Train the model with samples of normal and adversarial examples!
        model_train(sess, x, y, model_3, x_train_saliency, y_train_saliency,
                    evaluate=evaluate_saliency(),
                    args=train_params, rng=rng)

        #todo: use jsma to create adversarial testing??? or training???



    # Redefine TF model FGSM!!!
    model_2 = make_basic_cnn(nb_filters=nb_filters)
    preds_2 = model_2(x)
    fgsm2 = FastGradientMethod(model_2, sess=sess)

    # parameter for FGSM
    fgsm_params_y = {'eps': 0.3,
                     'y': y,
                     'clip_min': 0.,
                     'clip_max': 1.}
    adv_x_2 = fgsm2.generate(x, **fgsm_params_y)
    if not backprop_through_attack:
        # For the fgsm attack used in this tutorial, the attack has zero
        # gradient so enabling this flag does not change the gradient.
        # For some other attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adv_x_2 = tf.stop_gradient(adv_x_2)
    preds_2_adv = model_2(adv_x_2)

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

    ###########################################################################
    # MODEL Train for FGSM
    ###########################################################################
    # Perform and evaluate adversarial training with FSGM MODEL!!!
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, rng=rng)


    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy

    return report




def main(argv=None):
    minist_fgsm_saliency(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
    print("Tutorial Finished Running")