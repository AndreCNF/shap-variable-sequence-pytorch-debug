import torch                            # PyTorch to create and apply deep learning models
from torch import nn, optim             # nn for neural network layers and optim for training optimizers
from torch.nn import functional as F    # Module containing several activation functions

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers, p_dropout):
        super().__init__()

        self.n_inputs = n_inputs         # Number of input features
        self.n_hidden = n_hidden         # Number of hidden units
        self.n_outputs = n_outputs       # Number of outputs
        self.n_layers = n_layers         # Number of RNN layers
        self.p_dropout = p_dropout       # Probability of dropout

        # RNN layer(s)
        self.rnn = nn.RNN(self.n_inputs, self.n_hidden, self.n_layers, batch_first=True, dropout=self.p_dropout)

        # Fully connected layer which takes the RNN's hidden units and calculates the output classification
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)

        # Dropout used between the last RNN layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)

    def forward(self, x, x_lengths=None, get_hidden_state=False, hidden_state=None):
        # Get the batch size (might not be always the same)
        batch_size = x.shape[0]

        if hidden_state is None:
            # Reset the RNN hidden state. Must be done before you run a new batch. Otherwise the RNN will treat
            # a new batch as a continuation of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state

        if x_lengths is not None:
            # pack_padded_sequence so that padded items in the sequence won't be shown to the RNN
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        # Get the outputs and hidden states from the RNN layer(s)
        rnn_output, self.hidden = self.rnn(x, self.hidden)

        if x_lengths is not None:
            # Undo the packing operation
            rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        # Apply dropout to the last RNN layer
        rnn_output = self.dropout(rnn_output)

        # Flatten RNN output to fit into the fully connected layer
        flat_rnn_output = rnn_output.contiguous().view(-1, self.n_hidden)

        # Classification scores after applying the fully connected layers and softmax
        output = torch.sigmoid(self.fc(flat_rnn_output))

        if get_hidden_state:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels, x_lengths):
        # Before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # Flatten all the labels and make it have type long instead of float
        y_labels = y_labels.contiguous().view(-1).long()

        # Flatten all predictions
        y_pred = y_pred.view(-1, self.n_outputs)

        # Create a mask by filtering out all labels that are not a padding value
        # Also need to make it have type float to be able to multiply with y_pred
        mask = (y_labels <= 1).float()

        # Count how many predictions we have
        n_pred = int(torch.sum(mask).item())

        # Check if there's only one class to classify (either it belongs to that class or it doesn't)
        if self.n_outputs == 1:
            # Add a column to the predictions tensor with the probability of not being part of the
            # class being used
            y_pred_other_class = 1 - y_pred
            y_pred = torch.stack([y_pred_other_class, y_pred]).permute(1, 0, 2).squeeze()

        # Pick the values for the label and zero out the rest with the mask
        y_pred = y_pred[range(y_pred.shape[0]), y_labels * mask.long()] * mask
        # I need to get the diagonal of the tensor, which represents a vector of each
        # score (y_pred) multiplied by its correct mask value
        # Otherwise we get a square matrix of every score multiplied by every mask value

        # Completely remove the padded values from the predictions using the mask
        y_pred = torch.masked_select(y_pred, mask.byte())

        # Compute cross entropy loss which ignores all padding values
        ce_loss = -torch.sum(torch.log(y_pred)) / n_pred

        return ce_loss

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of RNN
        weight = next(self.parameters()).data

        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()

        if train_on_gpu:
            hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda()
        else:
            hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()

        return hidden
