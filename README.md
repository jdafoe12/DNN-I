# DNN-I

### About

DNN-I was developed both for learning and teaching purposes. Most importantly, my aim was to build a *concrete* understanding, of how deep neural networks are trained and how inference works. To achieve this, I implemented everything from scratch, using no special libraries. This gave me much freedom in language choice. I chose Guile Scheme for a couple reasons:
- I thought it would be a good opportunity to be my first project written in Guile Scheme. I am (slowly) working my way through *Structure and Interpretation of Computer Programs (SICP)* and wanted to apply some of the learned principles.
- Given the history of lisp as a language for artificial intelligence applications, I thought it was a rather natural choice.

For my first DNN, I chose to work with the MNIST dataset, inspired largely by the [3Blue1Brown's neural network video series](https://www.3blue1brown.com/topics/neural-networks). My initial target was to achieve 97% or higher accuracy, and have achieved ---.

In designing this code, I focused on enabling rapid experimentation with different hyperparameters, so they could be tweaked for optimal performance.

### Documentation

#### The Model
This is a standard fully connected deep neural network (DNN), which can be customized to have L layers. The entire model is initialized with random parameters (weights and biases) as follows:
```scheme
(define layers (list (initialize-layer 784 n_1)
					 (initialize-layer n_1 n_2)
					 .
                     .
                     .
					 (initialize-layer n_{L-1} n_L)))
```
where $n_i$ is the number of perceptrons (aka neurons, nodes, units) in layer $i$. The default number of layers is 4 (hence $L = 4$), and by default $n_1 = 32$, $n_2 = 32$, $n_3 = 32$. The output layer always has 10 perceptrons (hence $n_L = 10$), since MNIST is a classification task with 10 categories (digits 0-9).

The weights and biases are initialized using He initialization. Between each layer, the non-linear activation function ReLU is applied to the output perceptrons. In the output layer (layer $L$), Softmax is applied to convert logits into class probabilities for classification. The output is therefore a probability distribution over over the 10 MNIST categories.

#### Training

After initializing a model, it can be trained on the MNIST training set by calling
```scheme
(train-mnist layers epochs num-steps learning-rate)
```
where `layers` is defined [as above](#the-model), `epochs` is the number of training epochs, `num-steps` is the number of training steps per epoch, and `learning-rate` is the desired learning rate.

- During training, backpropagation updates the network’s weights by minimizing the cross-entropy loss through stochastic gradient descent, where each training step processes a single sample. 
- Each epoch will process `num-steps` training steps, and thus process `num-steps` training samples. At the beginning of an epoch, the *entire* MNIST training set is loaded into memory and permuted pseudo-randomly. Consequently, each step pseudo-randomly selects a training sample to process.
- After training completes, it can be written to a file by calling `(save-model layers filename)`. The default `filename` is "trained-model".

The procedures mentioned above are called by default within the file `training.scm`.

#### Testing

Testing the model is rather simple:
1. The entire MNIST training set is loaded into memory.
2. Inference is ran over each sample, and the model output is compared to the ground truth label.
3. The results are agregated to compute the model accuracy.

### Setup and Usage

#### Installing Guile

##### Ubuntu/Debian
Run
```sh 
apt-get install guile-3.0
```
##### Arch Linux
```sh 
pacman -S guile 
```

#### Extracting MNIST dataset

The MNIST dataset is included in this repository under the MNIST folder in compressed format. To extract these files, run 
```sh
gunzip MNIST/*.gz
```

#### Usage

To train the model, run
```sh 
guile training.scm 
```

To test the model, run 
```sh 
guile testing.scm 
```

Within `training.scm` hyperparameters may be modified by altering calls to the procedures [mentioned above](#documentation). If the output model file name is altered, make sure to update `testing.scm` accordingly.
