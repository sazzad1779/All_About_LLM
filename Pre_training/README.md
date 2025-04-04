# LLM Pre-Training Fundamentals

### There are three main types of Transformer-based models:
- Encoder-only: BERT,RoBERTa (good at understanding tasks like classification)
- Decoder-only: GPT,LLaMA, Falcon, Mistral, Claude (good at generating text)
- Encoder-Decoder: T5, BART (good for translation, summarization)

## Training Methods
#### Pretraining:

***Masked Language Modeling (MLM)***: Used by BERT, RoBert. Random words are masked and the model learns to predict them.
- Objective: Predict missing words in a sentence
- Input: "The cat [MASK] on the mat"
- Target: "sat"

***Causal Language Modeling (CLM)***: Used by GPT. The model predicts the next word in a sequence.
- Objective: Predict the next word
- Given: "The cat sat on the"
- Target: "mat"
- Training uses left-to-right attention only.

***Sequence-to-Sequence***: Used by T5, BART. Input-output pairs (e.g., translation: English → French).
