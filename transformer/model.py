import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        # d_model: The dimensionality of the embedding vector for each token.
        # vocab_size: The size of the vocabulary, i.e., the total number of unique tokens in the input data.
        self.d_model = d_model
        self.vocab_size = vocab_size
        # nn.Embedding is a PyTorch module that creates word embeddings for a given vocabulary.
        # It takes the vocab_size (number of tokens) and d_model (embedding dimension) as input arguments.
        self.embedding = nn.Embedding(vocab_size, d_model)
        # out

    def forward(self, x):
        # x: Input tensor representing a batch of token sequences.
        # The shape of x is (Batch_size, seq_len), where Batch_size is the number of samples in the batch,
        # and seq_len is the length of each input sequence (number of tokens in each sample).

        # The embedding layer maps each token to its dense vector representation.
        # The output of the embedding layer has shape (Batch_size, seq_len, d_model).
        embedded_input = self.embedding(x)

        # The embeddings are scaled according to the Transformer paper's recommendation.
        # The scaling factor is the square root of the embedding dimension (d_model).
        # This scaling helps in better training stability.
        scaled_embedded_input = embedded_input * math.sqrt(self.d_model)

        return scaled_embedded_input


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        # d_model: The dimensionality of the embedding vector for each token.
        # seq_len: The maximum sequence length of the input data. This is used to create positional encodings for each position in the sequence.
        # dropout: The dropout probability used to apply dropout to the positional encodings.

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) to hold the positional encodings.
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1) containing values from 0 to (seq_len - 1).
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # numerator

        # Create a vector of shape (d_model, 1) containing values corresponding to the exponent for each dimension.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # denominator in log space (more numerically stable)

        # Apply sine to even indices of the matrix to get the positional encoding for the even dimensions.
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))

        # Apply cosine to odd indices of the matrix to get the positional encoding for the odd dimensions.
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding to make it compatible with batched input.
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the positional encoding as a buffer to keep the tensor in the file of the model but not as a learned parameter.
        # Buffers won't be updated during training like parameters but will still be part of the model's state_dict.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: Input tensor representing a batch of token sequences.
        # The shape of x is (Batch_size, seq_len, d_model), where Batch_size is the number of samples in the batch,
        # seq_len is the length of each input sequence (number of tokens in each sample), and d_model is the embedding dimension.

        # Add the positional encoding to every word inside the sentence.
        # The positional encoding is added to the first `seq_len` positions of the input tensor x.
        positional_encoding = self.pe[:, :x.shape[1], :]
        x = x + positional_encoding.requires_grad_(False) # (Batch_size, seq_len, d_model)

        # Apply dropout to the output to improve regularization during training.
        x = self.dropout(x)

        return x


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None: # eps is for numerical stability
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied. alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # Added. bias is a learnable parameter

    def forward(self, x):
        x = x.float()
        
        # x: (Batch_size, seq_len, hidden_size)
        # Keep the dimension fro broadcasting (mean cancels the dimension when applied so dimension so flag needed)
        mean = x.mean(-1, keepdim=True) # (Batch_size, seq_len, 1)
        std = x.std(-1, keepdim=True)
        # eps is to prevent dividing by zero when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Create the first linear layer (fully connected) with input size d_model and output size d_ff.
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 & B1 (bias is True by default)
        
        # Create a dropout layer with dropout probability specified by the dropout argument.
        self.dropout = nn.Dropout(dropout)
        
        # Create the second linear layer (fully connected) with input size d_ff and output size d_model.
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 & B2 (bias is True by default)

    def forward(self, x):
        # x: Input tensor representing a batch of token sequences.
        # The shape of x is (Batch_size, seq_len, d_model), where Batch_size is the number of samples in the batch,
        # seq_len is the length of each input sequence (number of tokens in each sample), and d_model is the embedding dimension.
        
        # Apply the first linear layer to the input tensor.
        # The output shape is (Batch_size, seq_len, d_ff).
        output_linear_1 = self.linear_1(x)

        # Apply the ReLU activation function to the output of the first linear layer.
        # The ReLU activation function returns the element-wise maximum of 0 and the input tensor.
        # The output shape remains the same as (Batch_size, seq_len, d_ff).
        output_relu = torch.relu(output_linear_1)

        # Apply dropout to the output of the ReLU activation function.
        # Dropout sets a fraction of elements in the input tensor to zero randomly to improve regularization during training.
        # The output shape remains the same as (Batch_size, seq_len, d_ff).
        output_dropout = self.dropout(output_relu)

        # Apply the second linear layer to the output of the dropout layer.
        # The output shape is (Batch_size, seq_len, d_model), which matches the input shape.
        output_linear_2 = self.linear_2(output_dropout)

        # The final output of the FeedForwardBlock is the output of the second linear layer.
        return output_linear_2


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads

        # d_model must be divisible by h because we will split d_model into h heads
        assert d_model % h == 0, "d_model must be divisible by h"  # Check for validity

        self.d_k = d_model // h  # Dimension of vector seen by each head

        # Linear transformations for query, key, value, and output projections for each head
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=nn.Dropout):
        d_k = query.shape[-1]  # Get the last dimension of the query tensor

        # Applying the attention formula from the paper (scaled dot-product attention)
        # (Batch_size, h, seq_len, d_k) -> (Batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # order of execution:
        # 1. transpose(-2, -1) -> (Batch_size, h, seq_len, d_k) -> (Batch_size, h, d_k, seq_len)
        # 2. query @ key -> (Batch_size, h, seq_len, seq_len)
        # 3. / math.sqrt(d_k) -> (Batch_size, h, seq_len, seq_len)

        # Apply masking if provided to handle padded tokens or future tokens
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            # The large negative number becomes close to zero in the softmax function

        # Apply the softmax activation function along the last dimension to get attention weights
        # No change in shape: (Batch_size, h, seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply dropout to the attention weights if dropout is provided
        if dropout is not None:
            attention_weights = dropout(attention_weights)

        # Compute the weighted sum of the value tensor using the attention weights
        # (attention_weights @ value) -> (Batch_size, h, seq_len, seq_len) @ (Batch_size, h, seq_len, d_k)
        # attention is for visualization
        return (attention_weights @ value), attention_weights
        

    def forward(self, q, k, v, mask):
        # Apply linear transformations to get the query, key, and value tensors
        # (Batch_size, seq_len, d_model) -> (Batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the query, key, and value into h different parts (h = Number of heads)
        # We split d_model into h parts where each part is d_k in size
        # (Batch_size, seq_len, d_model) -> (Batch_size, seq_len, h, d_k) -> (Batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate the attention scores using the static method attention()
        # x is the weighted sum of the value tensor using attention weights
        # (Batch_size, h, seq_len, d_k), self.attention_scores has the attention weights
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Reshape and combine the heads to get the output tensor
        # .transpose() : (Batch_size, h, seq_len, d_k) -> (Batch_size, seq_len, h, d_k)
        # .view() : (Batch_size, seq_len, h, d_k) -> (Batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Apply the output linear transformation and return the output tensor
        # (Batch_size, seq_len, d_model) -> (Batch_size, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)    # Dropout layer with the specified dropout rate
        self.norm = LayerNormalization()      # LayerNormalization instance

    def forward(self, x, sublayer):
        # Apply the normalization layer to the input tensor 'x'
        normalized_input = self.norm(x)
        
        # Apply the sublayer (e.g., a feedforward block or a multi-head attention block) to the normalized input
        # The sublayer can be any neural network component that processes the input tensor
        intermediate_output = sublayer(normalized_input)
        
        # In the attention is all you need paper, sublayer is applied before the normalization layer.
        # It doesn't matter, both work fine.
        # In fact normalizing before applying the sublayer helps with numerical stability.

        # Apply dropout to the intermediate output
        dropout_output = self.dropout(intermediate_output)
        
        # Add the dropout output to the original input tensor (residual connection)
        # This is the essence of the residual connection: x + Dropout(Sublayer(LayerNormalization(x)))
        # The addition operation allows the intermediate outputs of the sublayer to flow directly to the output
        # and be added to the original input, bypassing the sublayer in case it does not capture useful information.
        # This helps the model to learn residual mappings (the difference between the input and output)
        # and facilitates training of deeper neural networks.
        output = x + dropout_output

        return output        
        # All of the above operations can be combined into a single line as follows:
        # return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        # Multi-Head Self-Attention Block
        self.self_attention_block = self_attention_block

        # Feed-Forward Block
        self.feed_forward_block = feed_forward_block

        # Residual Connections
        # Create two ResidualConnection instances with the specified dropout rate
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Step 1: Self-Attention Sublayer
        # Apply Multi-Head Self-Attention to the input 'x' with the mask 'src_mask' (encoder mask)
        # and pass the result through a ResidualConnection.
        # The ResidualConnection adds the input tensor 'x' to the output of the self-attention sublayer.
        # This is the first residual connection in the encoder block.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # Step 2: Feed-Forward Sublayer
        # Apply the Feed-Forward Block to the output of the self-attention sublayer and
        # pass the result through another ResidualConnection.
        # The ResidualConnection adds the input tensor 'x' to the output of the feed-forward sublayer.
        # This is the second residual connection in the encoder block.
        x = self.residual_connections[1](x, self.feed_forward_block)

        # Return the final output after both the self-attention and feed-forward sublayers.
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Store the list of Encoder blocks (layers)
        self.layers = layers

        # Layer normalization instance to be applied after all Encoder blocks
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        # Iterate through the Encoder blocks (layers) and apply each block's forward pass to the input tensor 'x'
        for layer in self.layers:
            x = layer(x, src_mask)

        # Apply layer normalization to the output tensor 'x'
        # Layer normalization is applied after processing all Encoder blocks.
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        # Store the self-attention block, cross-attention block, and feed-forward block for the DecoderBlock
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # Create a ModuleList to hold three ResidualConnection blocks (for three residual connections in the block)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Step 1: Self-Attention Sublayer
        # Apply the self-attention block to the input tensor 'x' with the mask 'tgt_mask' (decoder mask)
        # and pass the result through a ResidualConnection.
        # The ResidualConnection adds the input tensor 'x' to the output of the self-attention sublayer.
        # This is the first residual connection in the decoder block.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # Step 2: Cross-Attention Sublayer
        # Apply the cross-attention block to the output tensor from the self-attention block.
        # The query comes from the decoder ('x'), and the key and value come from the encoder ('encoder_output').
        # The cross-attention sublayer also takes the 'src_mask' (encoder mask) as input to prevent attending to padding tokens in the encoder output.
        # Pass the result through another ResidualConnection.
        # The ResidualConnection adds the input tensor 'x' to the output of the cross-attention sublayer.
        # This is the second residual connection in the decoder block.
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

        # Step 3: Feed-Forward Sublayer
        # Apply the Feed-Forward Block to the output of the cross-attention sublayer and
        # pass the result through another ResidualConnection.
        # The ResidualConnection adds the input tensor 'x' to the output of the feed-forward sublayer.
        # This is the third residual connection in the decoder block.
        x = self.residual_connections[2](x, self.feed_forward_block)

        # Return the final output of the DecoderBlock after three residual connections and respective attentions
        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Store the list of Decoder blocks (layers)
        self.layers = layers

        # Layer normalization instance to be applied after all Decoder blocks
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Iterate through the Decoder blocks (layers) and apply each block's forward pass to the input tensor 'x'.
        # The 'encoder_output' is passed as an additional argument for cross-attention in each block.
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Apply layer normalization to the output tensor 'x'.
        # Layer normalization is applied after processing all Decoder blocks.
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        # Create a linear projection layer that maps the input 'd_model' dimensional tensor to 'vocab_size' dimensional tensor.
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Apply the linear projection layer to the input tensor 'x', followed by a log-softmax operation along the last dimension (vocab_size).
        # The log-softmax operation converts the raw scores into log-probabilities, making it more numerically stable during training.
        # The shape changed from (batch_size, seq_len, d_model) to (batch_size, seq_len, vocab_size).
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()

        # Store the encoder, decoder, input embeddings, positional encodings, and projection layer
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # Step 1: Source Embedding
        # Convert the source input 'src' into embeddings using the src_embed layer.
        src = self.src_embed(src)

        # Step 2: Source Positional Encoding
        # Apply the positional encoding to the source embeddings.
        src = self.src_pos(src)

        # Step 3: Encoding
        # Pass the source embeddings through the encoder to obtain contextual representations.
        encoder_output = self.encoder(src, src_mask)

        # Return the encoder output, which contains the contextual representations of the source sequence.
        return encoder_output

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # Step 1: Target Embedding
        # Convert the target input 'tgt' into embeddings using the tgt_embed layer.
        tgt = self.tgt_embed(tgt)

        # Step 2: Target Positional Encoding
        # Apply the positional encoding to the target embeddings.
        tgt = self.tgt_pos(tgt)

        # Step 3: Decoding
        # Pass the target embeddings through the decoder along with encoder_output,
        # source mask 'src_mask', and target mask 'tgt_mask' to obtain decoder outputs.
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        # Return the decoder output, which contains the contextual representations of the target sequence.
        return decoder_output

    def project(self, x):
        # Step 1: Projection
        # Pass the input tensor 'x' through the projection_layer to perform the final classification.
        # This maps the contextual representations to the number of classes for the classification task.
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int = 512,
                      d_model: int=512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create embedding layers for the source and target vocabularies
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers for the source and target sequences
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create a list to hold encoder blocks
    encoder_blocks = []
    # Create 'N' encoder blocks and append them to the list
    for _ in range(N):
        # Create the self-attention block and feed-forward block for the encoder
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # Create an encoder block using the self-attention and feed-forward blocks
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        # Add the encoder block to the list
        encoder_blocks.append(encoder_block)

    # Create a list to hold decoder blocks
    decoder_blocks = []
    # Create 'N' decoder blocks and append them to the list
    for _ in range(N):
        # Create the self-attention block, cross-attention block, and feed-forward block for the decoder
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # Create a decoder block using the self-attention, cross-attention, and feed-forward blocks
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        # Add the decoder block to the list
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder using the ModuleList of encoder and decoder blocks
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer to map the contextual representations to the target vocabulary size
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer using the encoder, decoder, input embeddings, positional encodings, and projection layer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the weights of the transformer using Xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Return the built transformer
    return transformer
