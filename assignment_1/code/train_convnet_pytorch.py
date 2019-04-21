"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
    Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1))\
             .type(torch.FloatTensor).mean().item()
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy

def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # initialize empty dictionaries
    x, y, accu, loss = ({} for _ in range(4))

    # retrieve data
    data = cifar10_utils.get_cifar10(FLAGS.data_dir)

    # determine shapes
    image_shape = data['test'].images[0].shape
    nr_pixels = image_shape[0] * image_shape[1] * image_shape[2]
    nr_labels = data['test'].labels.shape[1]
    nr_test = data['test'].images.shape[0]

    # set standards
    tensor = torch.FloatTensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # save in variables
    for tag in data:
        nr_images = data[tag].images.shape[0]
        x_tmp = np.reshape(data[tag].images, (nr_images, nr_pixels))
        y_tmp = np.reshape(data[tag].labels, (nr_images, nr_labels))
        x[tag] = torch.tensor(x_tmp).type(tensor).to(device)
        y[tag] = torch.tensor(y_tmp).type(tensor).to(device)
        accu[tag] = []
        loss[tag] = []

    # create neural network
    neural_network = ConvNet(nr_pixels, dnn_hidden_units, nr_labels).to(device)
    cross_entropy = nn.CrossEntropyLoss().to(device)
    parameter_optimizer = torch.optim.Adam(neural_network.parameters(), \
                                 FLAGS.learning_rate)


    dx = 1
    i = 0
    logs = []
    while i < FLAGS.max_steps and np.linalg.norm(dx) > 1e-5:

        i += 1

        # sample batch from data
        rand_idx = np.random.randint(x['train'].shape[0], size=FLAGS.batch_size)
        x_batch = x['train'][rand_idx]
        y_batch = y['train'][rand_idx]

        parameter_optimizer.zero_grad()

        nn_out = neural_network.forward(x_batch)
        ce_out = cross_entropy.forward(nn_out, y_batch.argmax(dim=1))
        ce_out.backward()
        parameter_optimizer.step()

        if i % FLAGS.eval_freq == 0:

            # save train accuracy and loss
            accu['train'].append(accuracy(nn_out, y_batch))
            loss['train'].append(ce_out)

            # calculate and save test accuracy and loss
            nn_out = neural_network.forward(x['test'])
            ce_out = cross_entropy.forward(nn_out, y['test'].argmax(dim=1))
            accu['test'].append(accuracy(nn_out, y['test']))
            loss['test'].append(ce_out.item())

            # show results in command prompt and save log
            s = 'iteration ' + str(i) + ' | train acc/loss ' + \
                str('{:.3f}'.format(accu['train'][-1])) + '/' + \
                str('{:.3f}'.format(loss['train'][-1])) + ' | test acc/loss ' \
                + str('{:.3f}'.format(accu['test'][-1])) + '/' + \
                str('{:.3f}'.format(loss['test'][-1]))

            logs.append(s)
            print(s)
            #sys.stdout.write("\r%s" % s)
            #sys.stdout.flush()


    t = str(time.time())

    # write logs
    with open('results/logs_' + t + '.txt', 'w') as f:
        f.writelines(['%s\n' % item for item in logs])

    # write data to file
    with open('results/data_' + t + '.txt', 'w') as f:
        f.write('train accuracy')
        f.writelines([',%s' % str(item) for item in accu['train']])
        f.write('\ntrain loss')
        f.writelines([',%s' % str(item) for item in loss['train']])
        f.write('\ntest accuracy')
        f.writelines([',%s' % str(item) for item in accu['test']])
        f.write('\ntest loss')
        f.writelines([',%s' % str(item) for item in loss['test']])
    ########################
    # END OF YOUR CODE    #
    #######################

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
