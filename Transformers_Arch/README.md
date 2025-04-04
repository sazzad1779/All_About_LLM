# Transformers Architecture Fundamentals
![Transformers_Arch](attention_research_1.png)

The Transformer (from the 2017 paper “Attention Is All You Need”) is the backbone of most modern LLMs.
### What is a Transformer?

A Transformer is a deep learning model designed to handle sequential data, such as text, without relying on recurrence or convolution. It uses an innovative mechanism called self-attention, which allows the model to dynamically focus on different parts of the input data based on context.

This architecture forms the foundation of many state-of-the-art models like BERT, GPT, and T5.

Note:
Transformer-:

    1. Deep Learning Model
    2. Handle Sequential Data (Text)
    3. (alternative of)/ (dont relay on) recurrence or convolution.
    4. Use innovative mechanism (self-attention)
    5. self-attention allow model to dynamic focus on different part of input data based on context.


### Key Components of a Transformer:
1. Encoder:
> The encoder processes the input sequence and converts it into a dense, contextualized representation. 

> The encoder essentially extracts meaningful features from the input sequence, capturing the relationships and dependencies between tokens.
    
2. Decoder
> The decoder generates the output sequence (e.g., a translated sentence) by attending to both the encoder's output and previously generated tokens.

> The decoder predicts one token at a time, iteratively refining its output.

3. Self-Attention Mechanism
> The self-attention mechanism is the core innovation of Transformers. It computes relationships between all tokens in a sequence, allowing the model to focus on the most relevant parts.

> This mechanism enables the model to dynamically weigh tokens, capturing long-range dependencies effectively.
    
4. Multi-Head Attention
> Instead of calculating a single attention score, multi-head attention uses multiple attention "heads" to capture various aspects of the input relationships. Each head operates independently, enhancing the model's ability to understand complex patterns.
    
5. Positional Encoding
> Since Transformers process the input sequence as a whole (not sequentially like RNNs), positional encoding is used to inject information about the order of tokens.

> This encoding ensures the model understands the sequence order.
    
6. Feedforward Neural Network (FFN)
>   Each token's representation is independently processed through a feedforward neural network:
> - A linear transformation to increase dimensionality.
> - A non-linear activation function (e.g., ReLU).
> - Another linear transformation back to the original dimension.

7. Add & Norm
> The Add & Norm operation stabilizes the training process:

> - Add: Residual connections ensure gradient flow, preventing vanishing gradients.
> - Norm: Layer normalization standardizes intermediate outputs, improving model stability.

8. Masking
> Transformers employ masking to handle padding and sequential dependencies:
> - Padding Mask: Ignores padded tokens during attention calculations.
> - Look-Ahead Mask: Prevents the decoder from accessing future tokens during training.

9. Input Embedding
> Convert input ids to embedding vectors
