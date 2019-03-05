#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Char embeddings and decoder extracted from CS224N assignment 5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, NamedTuple

CharEmbeddingsShape = NamedTuple("CharEmbeddingsShape", [
    # size of one char embedding, for example 50 
    ('char_size', int),  
    # max numbers of chars in a word, for example 21
    ('chars_per_word', int), 
    # output embedding size
    ('out_size', int), 
    # number of chars in the input vocabulary
    ('vocab_size', int), 
    # the width of the cnn kernel, for example 5
    ('cnn_kernel', int),
    # idx of the pad token
    ('pad_token_idx', int),
    ('dropout_prob', float)
])

class CNN(nn.Module):
    """Character CNN
    """

    def __init__(self, in_channel: int, out_channels: int, kernel_size:int):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv1d(input)

    def initializeUniform(self, value: float):
        with torch.no_grad():
            self.conv1d.weight.data.fill_(value)
            self.conv1d.bias.data.fill_(0.0)


class Highway(nn.Module):
    """Highway Networks, Srivastava et al., 2015 https://arxiv.org/abs/1505.00387
    """

    def __init__(self, in_features: int, out_features: int, has_relu:bool=True):
        """
        Create the primitives used to build the Highway network.

        @param in_features: size of each input sample 
        @param out_features: size of the output sample
        """
        super(Highway, self).__init__()
        self.projLinear = nn.Linear(in_features=in_features, out_features=out_features)
        self.gateLinear = nn.Linear(in_features=in_features, out_features=out_features)
        self.has_relu = has_relu
         

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the input through the highway.
        """
        x_proj = self.projLinear(input)
        x_proj = F.relu(x_proj) if self.has_relu else x_proj
        x_gate = torch.sigmoid(self.gateLinear(input))
        return x_gate * x_proj + (1-x_gate) * input

    def initializeWeights(self, projection:Optional[float], gate:float):
        """Initialize all the weights in the projection and gate level to the same value.
        """
        with torch.no_grad():
            # initialize the projection layer
            if projection is None:
                torch.nn.init.xavier_uniform(self.projLinear.weight)
            else:
                self.projLinear.weight.data.fill_(projection)
                self.projLinear.bias.data.fill_(0.0)

            # initialize the gate layer
            self.gateLinear.weight.data.fill_(gate)
            self.gateLinear.bias.data.fill_(0.0)

class CharEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, shape: CharEmbeddingsShape):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(CharEmbeddings, self).__init__()

        self.embed_size = shape.out_size
        self.embeddings = nn.Embedding(shape.vocab_size, shape.char_size, padding_idx=shape.pad_token_idx)
        self.cnn = CNN(in_channel=shape.char_size, out_channels=shape.out_size, kernel_size=shape.cnn_kernel)
        self.maxpool = nn.MaxPool1d(shape.chars_per_word-shape.cnn_kernel+1)
        self.highway = Highway(in_features=shape.out_size, out_features=shape.out_size)
        self.dropout = nn.Dropout(shape.dropout_prob)


    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        x1 = self.embeddings(input)
        x1_shaped = x1.view(x1.shape[0]*x1.shape[1], x1.shape[2], x1.shape[3])
        x1t =  x1_shaped.transpose(1,2)

        x2 = self.cnn(x1t)
        x_conv_out = self.maxpool(F.relu(x2))
        x3 = self.highway(x_conv_out.squeeze())
        x4 = self.dropout(x3)

        return x4.view(x1.shape[0], x1.shape[1], -1)


CharDecoderShape = NamedTuple("CharDecoderShape", [

    ('hidden_size', int), 
    # size of the vector in one char embedding, for example 50
    ('char_emb_size', int),
    ('padding_char_idx', int),
    ('output_vocab_size', int)
])

class CharDecoder(nn.Module):
    def __init__(self, target_vocab, shape: CharDecoderShape):
        """ Init Character Decoder.

        @param shape (CharDecoderShape): shape of the decoder, see CharDecoderShape
        """
        super(CharDecoder, self).__init__() 
        assert (shape.output_vocab_size == len(target_vocab.char2id)), "Wrong vocab size, expected {len(target_vocab.char2id)} but got {shape.output_vocab_size}"
        self.charDecoder = nn.LSTM(input_size=shape.char_emb_size, hidden_size=shape.hidden_size)
        self.char_output_projection = nn.Linear(out_features=shape.output_vocab_size, in_features=shape.hidden_size)
        self.decoderCharEmb = nn.Embedding(shape.output_vocab_size, shape.char_embed_size, padding_idx=shape.padding_char_idx)
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduction='sum', ignore_index=shape.padding_char_idx)
        self.target_vocab = target_vocab

    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        embeddings = self.decoderCharEmb(input)
        output, hidden = self.charDecoder(embeddings, dec_hidden) 
        s_t = self.char_output_projection(output)
        return (s_t, hidden)
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        input = char_sequence.narrow(0,0,char_sequence.shape[0]-1)
        target = char_sequence.narrow(0,1,char_sequence.shape[0]-1)
        output, dec_hidden = self.forward(input, dec_hidden)
        loss = 0.0
        for b in range(target.size(0)):
            loss += self.crossEntropyLoss(output[b, :, :], target[b, :])
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        batch_size =  initialStates[0].size(1)
        words = [""] * batch_size
        current = torch.tensor([self.target_vocab.start_of_word] * batch_size, device=device)

        # run decode loops until you reach the end of the max word
        for idx_in_word in range(0,max_length):
            s_t, initialStates = self(current.unsqueeze(0), initialStates)
            v,arg_max = F.softmax(s_t, dim=-1).max(dim=-1)
            for b, index in enumerate(arg_max[0]):
                current[b] = index
                if len(words[b]) < idx_in_word:
                    # this word is already complete
                    continue
                index = index.item()
                if index == self.target_vocab.end_of_word:
                    # this is the end of word, do not add
                    # complete words will start to fall behind
                    continue
                words[b] += self.target_vocab.id2char[index]

        return words

