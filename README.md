# MNIST-DNN
Training on MNIST digits 0-4 with DNN
# MNIST-DNN
Training on MNIST digits 0-4 with DNN
## Training Process
**1. Extract train/validation/test data from mnist**
**2. Build execution graph**
* He initialization 
* ELU activation function
* Adam optimization
* Five hidden layers with 128 neurons
* Output layer using linear activation
* Softmax layer for classification

**3. Compute loss**
* We use sparse_softmax_cross_entropy_with_logits to  compute the loss between logits from the output layer and labels from input data.
* The average loss will be minimize by optimizer.

**4. Compute accuracy**
* We use top_k function to find predicted labels from the softmax layer.
* The accuracy is computed by comparing with predicted and true labels.

**5. Training**
* Shuffle
    - Use np.random.shuffle to shuffle the index of training data.
* Batch
    - Use np.array_split to divide into batches. Each epoch will train all batches one by one.
    * Early stop
        - If the loss of the epoch is smaller than best_loss, we update the checkpoints.
        - If the loss of the epoch isn't smaller than best_loss, the patience of early stop will be cumulated. As the patience reaching threshold, the training process will be finished.

**6. Testing**
* Accuracy
    - The validation accuracy can obtain from model with validation data.
* Precision / Recall
    - Use the function in sklearn.metrics

**7. Bonus**
* Cross validation
    - We use the 10-fold cross validation to find hyperparameters, ex. learning rate, dropout ratio
* Dropout
    - We add tf.nn.dropout between each hidden layers. However, the improvement isn't very obvious.

## Hyperparameter
Finally, we use:
* learning_rate = 0.01
* batch_size = 2048
* dropout = 0.5
* early_stop = 20
