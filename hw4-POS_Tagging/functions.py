class BasicPOSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BasicPOSTagger, self).__init__()
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # You are required to create a model with:
        # an embedding layer: that maps words to the embedding space
        # an LSTM layer: that takes word embeddings as input and outputs hidden states
        # a Linear layer: maps from hidden state space to tag space
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, sentence):
        tag_scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        # Given a tokenized index-mapped sentence as the argument,
        # compute the corresponding scores for tags
        # returns:: tag_scores (Tensor)
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return tag_scores

def train(epoch, model, loss_function, optimizer):
    train_loss = 0
    train_examples = 0
    for sentence, tags in training_data:
        #############################################################################
        # TODO: Implement the training loop
        # Hint: you can use the prepare_sequence method for creating index mappings
        # for sentences. Find the gradient with respect to the loss and update the
        # model parameters using the optimizer.
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    avg_train_loss = train_loss / train_examples
    avg_val_loss, val_accuracy = evaluate(model, loss_function, optimizer)

    print("Epoch: {}/{}\tAvg Train Loss: {:.4f}\tAvg Val Loss: {:.4f}\t Val Accuracy: {:.0f}".format(epoch,
                                                                      EPOCHS,
                                                                      avg_train_loss,
                                                                      avg_val_loss,
                                                                      val_accuracy))

def evaluate(model, loss_function, optimizer):
  # returns:: avg_val_loss (float)
  # returns:: val_accuracy (float)
    val_loss = 0
    correct = 0
    val_examples = 0
    with torch.no_grad():
        for sentence, tags in val_data:
            #############################################################################
            # TODO: Implement the evaluate loop
            # Find the average validation loss along with the validation accuracy.
            # Hint: To find the accuracy, argmax of tag predictions can be used.
            #############################################################################

            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################
    val_accuracy = 100. * correct / val_examples
    avg_val_loss = val_loss / val_examples
    return avg_val_loss, val_accuracy

def test():
    val_loss = 0
    correct = 0
    val_examples = 0
    predicted_tags = []
    with torch.no_grad():
        for sentence in test_sentences:
            #############################################################################
            # TODO: Implement the test loop
            # This method saves the predicted tags for the sentences in the test set.
            # The tags are first added to a list which is then written to a file for
            # submission. An empty string is added after every sequence of tags
            # corresponding to a sentence to add a newline following file formatting
            # convention, as has been done already.
            #############################################################################

            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################
            predicted_tags.append("")

    with open('test_labels.txt', 'w+') as f:
        for item in predicted_tags:
            f.write("%s\n" % item)

#############################################################################
# TODO: Generate predictions from val data
# Create lists of words, tags predicted by the model and ground truth tags.
#############################################################################
def generate_predictions(model, test_sentences):
    # returns:: word_list (str list)
    # returns:: model_tags (str list)
    # returns:: gt_tags (str list)
    # Your code here
    return word_list, model_tags, gt_tags

#############################################################################
# TODO: Carry out error analysis
# From those lists collected from the above method, find the
# top-10 tuples of (model_tag, ground_truth_tag, frequency, example words)
# sorted by frequency
#############################################################################
def error_analysis(word_list, model_tags, gt_tags):
    # returns: errors (list of tuples)
    # Your code here
    return errors

class CharPOSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, char_embedding_dim,
                 char_hidden_dim, char_size, vocab_size, tagset_size):
        super(CharPOSTagger, self).__init__()
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # You are required to create a model with:
        # an embedding layer: that maps words to the embedding space
        # an char level LSTM: that finds the character level embedding for a word
        # an LSTM layer: that takes the combined embeddings as input and outputs hidden states
        # a Linear layer: maps from hidden state space to tag space
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, sentence, chars):
        tag_scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        # Given a tokenized index-mapped sentence and a character sequence as the arguments,
        # find the corresponding scores for tags
        # returns:: tag_scores (Tensor)
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return tag_scores

def train_char(epoch, model, loss_function, optimizer):
    train_loss = 0
    train_examples = 0
    for sentence, tags in training_data:
        #############################################################################
        # TODO: Implement the training loop
        # Hint: you can use the prepare_sequence method for creating index mappings
        # for sentences as well as character sequences. Find the gradient with
        # respect to the loss and update the model parameters using the optimizer.
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    avg_train_loss = train_loss / train_examples
    avg_val_loss, val_accuracy = evaluate_char(model, loss_function, optimizer)

    print("Epoch: {}/{}\tAvg Train Loss: {:.4f}\tAvg Val Loss: {:.4f}\t Val Accuracy: {:.0f}".format(epoch,
                                                                      EPOCHS,
                                                                      avg_train_loss,
                                                                      avg_val_loss,
                                                                      val_accuracy))

def evaluate_char(model, loss_function, optimizer):
    # returns:: avg_val_loss (float)
    # returns:: val_accuracy (float)
    val_loss = 0
    correct = 0
    val_examples = 0
    with torch.no_grad():
        for sentence, tags in val_data:
            #############################################################################
            # TODO: Implement the evaluate loop
            # Find the average validation loss along with the validation accuracy.
            # Hint: To find the accuracy, argmax of tag predictions can be used.
            #############################################################################

            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################
    val_accuracy = 100. * correct / val_examples
    avg_val_loss = val_loss / val_examples
    return avg_val_loss, val_accuracy

#############################################################################
# TODO: Generate predictions from val data
# Create lists of words, tags predicted by the model and ground truth tags.
#############################################################################
def generate_predictions_char(model, test_sentences):
    # returns:: word_list (str list)
    # returns:: model_tags (str list)
    # returns:: gt_tags (str list)
    # Your code here
    return word_list, model_tags, gt_tags

#############################################################################
# TODO: Carry out error analysis
# From those lists collected from the above method, find the
# top-10 tuples of (model_tag, ground_truth_tag, frequency, example words)
# sorted by frequency
#############################################################################
def error_analysis_char(word_list, model_tags, gt_tags):
    # returns: errors (list of tuples)
    # Your code here
    return errors
