### Introduction and Plan of Action

Hello all, my name is Krish Naik, and welcome to my YouTube channel. In this video, we'll delve into a one-shot, complete, in-depth mathematical intuition about Transformers[cite: 2]. I know many of you were specifically waiting for this video. This is a 5-hour tutorial, and along with it, I'll provide complete materials with in-depth mathematical intuition, examples, and diagrams[cite: 3]. Go ahead and enjoy this session, make sure to write your notes, share them on LinkedIn, and tag me. I'll try to check in on your learning process[cite: 3]. Let's enjoy this session[cite: 4].

We are going to continue the discussion regarding NLP with deep learning, focusing on Transformers in this video and the upcoming series[cite: 4]. Transformers are a very important topic if you want to excel in deep learning, specifically for NLP tasks[cite: 5]. In this video, I'll outline the plan of action for covering this topic. We've already discussed RNN, LSTM, GRU, understood their problems, moved to encoder-decoder architecture in sequence-to-sequence learning, and addressed issues there with the attention mechanism[cite: 5, 6]. Each architecture had differences. Now, we'll understand the problems in the attention mechanism and how Transformers solve them[cite: 6].

The plan is: first, understand *why* Transformers. Then, we'll examine the Transformer architecture[cite: 6]. The architecture (shown on the right) looks complex initially, with many components different from encoder-decoder or attention mechanisms[cite: 7]. We'll break it down. It has an encoder and a decoder, but with many more internal components[cite: 7]. We'll start by understanding why Transformers, then move to the architecture, covering the first module: **self-attention**[cite: 8]. In self-attention, you'll see Q, K, V parameters (Query, Key, Value pairs), which we'll discuss[cite: 9]. Then, we'll cover **positional encoding**, another crucial topic[cite: 9, 10]. Next is **multi-head attention**, and finally, we'll combine these to understand the working of Transformers[cite: 10].

Transformers are vital because most current generative AI models, including large language models (LLMs) and multimodal models, are trained using the Transformer architecture[cite: 10, 11]. Examples include BERT and GPT. OpenAI's models like ChatGPT, currently using GPT-4o, are based on this architecture, trained with vast amounts of data[cite: 11]. GPT-4 uses the GPT architecture with transfer learning on this architecture, trained with huge data[cite: 12]. We'll cover these topics. In the next video, I'll discuss *why* Transformers, revisiting the encoder-decoder architecture, attention mechanism, and their problems[cite: 12, 13].

### Definition and Applications of Transformers

Let's continue with Transformers. The first topic is understanding *what* and *why*. A simple definition: Transformers in natural language processing are a type of deep learning model using a **self-attention mechanism** to analyze and process natural language data[cite: 13, 14]. They are encoder-decoder models used for applications like machine translation[cite: 15]. Machine translation is a sequence-to-sequence task[cite: 16].

In sequence-to-sequence tasks, like language translation (e.g., English to French using Google Translate), the input (English sentence) consists of many words, and the output (French sentence) also consists of many words. This is a many-to-many sequence-to-sequence task[cite: 17, 18]. The length of the sentences matters, and as it increases, we need good accuracy[cite: 18]. Transformers are commonly used for such tasks[cite: 19].

### Limitations of Encoder-Decoder Architectures

You might recall that the previous encoder-decoder architecture was also used for sequence-to-sequence tasks[cite: 19]. In that architecture, we had an encoder and a decoder, often using LSTMs[cite: 19]. Words (X1, X2, X3) were input sequentially based on timestamps (t=1, t=2, t=3) after being converted into vectors via an embedding layer[cite: 20, 21]. The encoder processed these inputs without producing intermediate outputs, finally generating a single context vector (C vector) representing the entire input sentence[cite: 21, 22]. This context vector was then fed into the decoder (also often an LSTM) to generate the output sequence, using techniques like softmax for prediction[cite: 22, 23].

The attention mechanism improved upon the basic encoder-decoder by providing additional context to the decoder at each step, enhancing performance, especially for longer sentences[cite: 24]. Bidirectional LSTMs/RNNs were often used in the encoder part of attention models[cite: 24]. However, a significant problem with both standard encoder-decoder and attention mechanisms is that words are processed sequentially based on timestamps (t=1, t=2, etc.)[cite: 25]. This means we **cannot send all words in a sentence in parallel**[cite: 25]. This sequential processing makes these models **not scalable** for training on very large datasets[cite: 26].

### Advantage of Self-Attention

Transformers address this scalability issue[cite: 27]. They **do not use LSTMs or RNNs** in their encoders or decoders[cite: 27]. Instead, they use a **self-attention module**[cite: 27]. Because of self-attention, **all words in the input sequence are processed in parallel**[cite: 27]. You can see this in the architecture diagram where input embeddings are processed simultaneously[cite: 27]. This parallelism is a key reason why Transformers are so effective and scalable[cite: 28]. To handle the order of words when processing in parallel, Transformers use **positional encoding**, which we'll discuss later[cite: 28]. The self-attention module computes the relationships between words in the sentence, allowing for parallel processing[cite: 28].

### Benefits of Transformers: Scalability and Transfer Learning

The inability to process words in parallel makes attention mechanisms and encoder-decoders less scalable[cite: 29]. Transformers, being scalable, perform well even as the dataset size increases significantly, leading to state-of-the-art (SOTA) models in NLP[cite: 30]. This scalability has been crucial for training models like BERT and GPT on huge datasets[cite: 32, 33].

Furthermore, Transformer applications are not restricted to NLP. Through **transfer learning**, they are now used in **multimodal tasks** involving both NLP and images, performing exceptionally well[cite: 31, 33]. For example, OpenAI applications like DALL-E, which generate images from text descriptions, are based on Transformers[cite: 34]. This architecture is fundamental to many LLMs used in generative AI[cite: 34].

The key takeaway is the parallel processing capability enabled by the self-attention module, overcoming the sequential limitations of RNN-based encoder-decoders and attention mechanisms[cite: 35].

### Contextual Embeddings

Another important problem addressed by Transformers is related to **contextual embeddings**[cite: 36]. Let's consider the sentence: "My name is Krish and I play cricket"[cite: 36]. In traditional embedding layers (like word2vec), each word ("My", "name", "is", "Krish", etc.) gets a fixed vector representation, regardless of the context[cite: 37]. While encoder-decoders used these fixed embeddings, Transformers, via self-attention, generate contextual embeddings[cite: 38]. This means the vector for a word like "Krish" will incorporate information about the other words in the sentence ("My", "name", "is", "and", "I", "play", "cricket")[cite: 38]. This ability to create contextual embeddings is crucial for understanding nuanced meanings in language[cite: 38].

### Transformer Architecture Overview

Let's delve into the architecture shown on the right, which might seem complex[cite: 39]. Instead of looking at the full diagram directly, we'll break it down step-by-step, following the plan: self-attention, positional encoding, multi-head attention, etc.[cite: 39].

A basic Transformer block can be used for sequence-to-sequence tasks like language translation (e.g., English to French)[cite: 40, 41]. The input is the English sentence, and the output is the French sentence[cite: 41]. Inside this block are **encoders** and **decoders**, following an encoder-decoder architecture[cite: 42].

However, it's not just one encoder and one decoder. The Transformer uses a stack of multiple identical encoders and decoders[cite: 43]. The input text goes through the stack of encoders sequentially, from bottom to top[cite: 43, 45]. The famous "Attention Is All You Need" research paper, which introduced Transformers, used **N=6 encoders** and **N=6 decoders** stacked on top of each other[cite: 44, 46, 47]. The output from the top encoder is then passed to all the decoders in the stack[cite: 45, 46]. The decoder stack also processes information sequentially, ultimately producing the output sequence (e.g., the French translation)[cite: 46].

### Inside the Encoder and Decoder

What's inside a single encoder layer? Each encoder consists of two main sub-layers:
1.  A **self-attention mechanism**[cite: 48].
2.  A **feed-forward neural network**[cite: 48].

The input goes through the self-attention layer first, then the feed-forward network[cite: 49].

What's inside a single decoder layer? Each decoder has three main sub-layers:
1.  A **masked self-attention mechanism** (we'll see why it's masked later)[cite: 49].
2.  An **encoder-decoder attention mechanism** (which attends to the output of the encoder stack)[cite: 50].
3.  A **feed-forward neural network**[cite: 49, 50].

Now, let's focus on what the **self-attention** layer does in the encoder[cite: 50]. Suppose the input sentence is "How are you"[cite: 51]. These words are first converted into input vectors (embeddings)[cite: 51]. The self-attention layer takes these vectors and transforms them into **contextual vectors**[cite: 55].

### Contextual Vectors and Parallel Processing

What's the difference between the initial fixed vectors and the contextual vectors produced by self-attention? The initial vectors (e.g., from an embedding layer like word2vec) are fixed for each word[cite: 56]. The contextual vector for a word, say "How", generated by self-attention, is influenced by the other words in the sequence ("are", "you")[cite: 56, 57]. It captures the context[cite: 57]. Similarly, the vector for "are" will depend on "How" and "you", and the vector for "you" will depend on "How" and "are"[cite: 58].

Why are contextual vectors needed? They help the model understand the meaning better, especially in longer sentences, leading to improved accuracy[cite: 59]. Crucially, as mentioned before, the self-attention mechanism allows **all words to be processed in parallel**, unlike the sequential processing in RNNs/LSTMs used in older architectures[cite: 59, 60]. This parallelization makes the Transformer architecture highly scalable and efficient for training on large datasets[cite: 60].

The output vectors (contextual vectors) from the self-attention layer are then passed to the feed-forward neural network within the same encoder layer[cite: 51, 52]. The output of the feed-forward network from one encoder layer becomes the input to the next encoder layer in the stack[cite: 52, 53]. This process repeats through all N encoder layers[cite: 53, 54].

**Summary of Encoder Flow:** Input sentence -> Input Embedding + Positional Encoding -> Multi-Head Self-Attention -> Add & Norm (Residual Connection + Layer Normalization) -> Feed-Forward Network -> Add & Norm -> Output to next Encoder or Decoder stack[cite: 54, 55].

### Self-Attention Deep Dive: Query, Key, and Value Vectors

Let's discuss self-attention in more detail[cite: 61]. How does it convert fixed input vectors into contextual vectors[cite: 62]? The core idea involves calculating attention scores based on how relevant words are to each other.

First, we need to get initial token embeddings. We can use standard embedding layers for this (e.g., converting "The", "cat", "sat" into fixed dense vectors)[cite: 63]. The challenge is converting these fixed vectors into contextual vectors dynamically based on the sentence[cite: 64].

The first step within self-attention is to derive three vectors from each input embedding:
1.  **Query vector (Q)**
2.  **Key vector (K)**
3.  **Value vector (V)** [cite: 65]

These vectors are created for *every* input token (word)[cite: 66]. What do they represent?
* **Query Vector (Q):** Represents the current token for which we are calculating attention. It's like asking a question: "What parts of the sentence are relevant to me?"[cite: 68].
* **Key Vector (K):** Represents the token as a potential "match" for a query. It's like a label or identifier for the word's information.
* **Value Vector (V):** Represents the actual information or content of the token. If a query matches a key, the corresponding value is what gets passed along.

How are Q, K, and V created? They are generated through **linear transformations** of the input embedding vector[cite: 69]. We multiply the input embedding vector (let's call it `x`) by three distinct weight matrices that are learned during training: `Wq`, `Wk`, and `Wv`[cite: 70, 71].

* `q = x * Wq`
* `k = x * Wk`
* `v = x * Wv` [cite: 71, 72, 73, 74]

These weight matrices (`Wq`, `Wk`, `Wv`) are initially randomized and then learned via backpropagation during the training process[cite: 74].

**Example Calculation:**
Let's say we have the sentence "The cat sat" and use simplified 4-dimensional identity matrices for `Wq`, `Wk`, `Wv` and simple vectors for the words just for illustration:
* `embedding(The)` = `[1, 0, 1, 0]`
* `embedding(cat)` = `[0, 1, 0, 1]`
* `embedding(sat)` = `[1, 1, 1, 1]`

Since `Wq`, `Wk`, `Wv` are identity matrices in this simplified case:
* `Q(The) = K(The) = V(The) = [1, 0, 1, 0]`
* `Q(cat) = K(cat) = V(cat) = [0, 1, 0, 1]` [cite: 75]
* `Q(sat) = K(sat) = V(sat) = [1, 1, 1, 1]` [cite: 75, 76]

In reality, `Wq`, `Wk`, `Wv` are learned, and the dimensions are much larger.

### Calculating Attention Scores

The next step is to compute **attention scores**[cite: 77]. The score determines how much attention a word should pay to other words (including itself) in the sentence. It's calculated by taking the **dot product** of the **query vector (Q)** of the word we're focusing on, with the **key vector (K)** of every word in the sentence[cite: 77].

`Score(word_i, word_j) = Q(word_i) . K(word_j)` [cite: 77]

This dot product measures the similarity or compatibility between the query and the key. A higher score means word `j` is more relevant to word `i`[cite: 78].

**Example Calculation (for the word "The"):**
We need to calculate the score for "The" with respect to "The", "cat", and "sat".

1.  **Score(The, The):**
    * `Q(The) . K(The)^T = [1, 0, 1, 0] . [1, 0, 1, 0]^T`
    * `= (1*1) + (0*0) + (1*1) + (0*0) = 2` [cite: 79, 80]
2.  **Score(The, cat):**
    * `Q(The) . K(cat)^T = [1, 0, 1, 0] . [0, 1, 0, 1]^T`
    * `= (1*0) + (0*1) + (1*0) + (0*1) = 0` [cite: 80, 81]
3.  **Score(The, sat):**
    * `Q(The) . K(sat)^T = [1, 0, 1, 0] . [1, 1, 1, 1]^T`
    * `= (1*1) + (0*1) + (1*1) + (0*1) = 2` [cite: 82]

So, for the word "The", the attention scores are `[2, 0, 2]` corresponding to ("The", "cat", "sat"). This suggests "The" pays attention to itself and "sat", but not "cat" in this simplified example[cite: 83, 84].

We repeat this process for every word in the sentence:
* **For "cat":** Scores = `[Score(cat, The), Score(cat, cat), Score(cat, sat)] = [0, 2, 2]` [cite: 85, 86, 87]
* **For "sat":** Scores = `[Score(sat, The), Score(sat, cat), Score(sat, sat)] = [2, 2, 4]` [cite: 88, 89]

### Scaling the Scores

The next step is **scaling** the attention scores[cite: 90]. Why? If the dot products become very large, applying the softmax function (next step) can lead to extremely small gradients during backpropagation, making training unstable (vanishing gradient problem)[cite: 96, 97]. Large scores pushed through softmax tend to result in probabilities very close to 0 or 1, concentrating all the attention weight on one token[cite: 95, 96].

To mitigate this, the scores are scaled down by dividing them by the square root of the dimension of the key vectors (`dk`)[cite: 99, 100].

`Scaled Score = Score / sqrt(dk)` [cite: 99]

In the "Attention Is All You Need" paper, `dk` was 64, so the scaling factor was `sqrt(64) = 8`[cite: 100, 186]. In our simple example, let's assume `dk` = 4 (dimension of our K vectors), so we divide by `sqrt(4) = 2`[cite: 100, 111].

**Example Calculation (Scaled Scores):**
* **For "The":** `[2/2, 0/2, 2/2] = [1, 0, 1]` [cite: 113, 114]
* **For "cat":** `[0/2, 2/2, 2/2] = [0, 1, 1]`
* **For "sat":** `[2/2, 2/2, 4/2] = [1, 1, 2]`

Scaling helps stabilize training and produces more balanced attention weights[cite: 105, 107, 108].

### Applying Softmax

After scaling, the **softmax function** is applied to the scaled scores for each word[cite: 114, 115]. Softmax converts the scores into probabilities (attention weights) that sum up to 1. These weights represent the distribution of attention the current word gives to all words in the sentence.

`Attention Weights = softmax(Scaled Scores)` [cite: 115]

**Example Calculation (Attention Weights):**
Let's apply softmax to the scaled scores for "The": `[1, 0, 1]`

* `softmax([1, 0, 1])`
* `= [exp(1)/(exp(1)+exp(0)+exp(1)), exp(0)/(exp(1)+exp(0)+exp(1)), exp(1)/(exp(1)+exp(0)+exp(1))]`
* `≈ [2.718 / (2.718+1+2.718), 1 / (2.718+1+2.718), 2.718 / (2.718+1+2.718)]`
* `≈ [2.718 / 6.436, 1 / 6.436, 2.718 / 6.436]`
* `≈ [0.422, 0.155, 0.422]` (These are the attention weights for "The") [cite: 116]

Similarly for other words:
* **For "cat":** `softmax([0, 1, 1]) ≈ [0.155, 0.422, 0.422]` [cite: 118]
* **For "sat":** `softmax([1, 1, 2]) ≈ [0.2119, 0.2119, 0.5762]` [cite: 118, 119, 120]

### Weighted Sum of Values

The final step in the self-attention calculation is to compute the output vector for each word[cite: 120]. This is done by taking a **weighted sum of the value vectors (V)** of all words in the sentence, using the calculated attention weights as the weights[cite: 121].

`Output Vector(word_i) = Σ (Attention Weight(word_i, word_j) * V(word_j))` for all `j` in the sentence[cite: 121, 122].

**Example Calculation (Output Vector for "The"):**
* `Output(The) = AttentionWeight(The, The)*V(The) + AttentionWeight(The, cat)*V(cat) + AttentionWeight(The, sat)*V(sat)`
* `= 0.422 * [1, 0, 1, 0] + 0.155 * [0, 1, 0, 1] + 0.422 * [1, 1, 1, 1]`
* `= [0.422, 0, 0.422, 0] + [0, 0.155, 0, 0.155] + [0.422, 0.422, 0.422, 0.422]`
* `= [0.844, 0.577, 0.844, 0.577]` (This is the final contextual vector for "The" from this attention head) [cite: 122]

This process is repeated for "cat" and "sat" to get their respective contextual output vectors[cite: 125]. The resulting output vectors now incorporate contextual information from the entire sentence based on the attention mechanism.

**Summary of Self-Attention Steps:**
1.  Create Q, K, V vectors from input embeddings using learned weight matrices (`Wq`, `Wk`, `Wv`)[cite: 65, 70, 123].
2.  Calculate attention scores: `Score = Q * K^T`[cite: 77, 123].
3.  Scale scores: `Scaled Score = Score / sqrt(dk)`[cite: 99, 123].
4.  Apply Softmax: `Attention Weights = softmax(Scaled Scores)`[cite: 115, 124].
5.  Compute output vector: `Output = Attention Weights * V`[cite: 121, 124, 125].

### Multi-Head Attention

Instead of performing self-attention just once with one set of `Wq`, `Wk`, `Wv` matrices, Transformers use **Multi-Head Attention**[cite: 130]. This involves running the self-attention process multiple times in parallel, each with different, independently learned weight matrices (`Wq`, `Wk`, `Wv`)[cite: 131, 134]. Each parallel run is called an "attention head"[cite: 133].

Let's say we use 8 attention heads (as in the original paper [cite: 177]). For each input word, we would compute 8 different sets of Q, K, V vectors using 8 different sets of weight matrices (`Wq0, Wk0, Wv0`, `Wq1, Wk1, Wv1`, ..., `Wq7, Wk7, Wv7`)[cite: 135, 137, 138]. Each head performs the scaling, softmax, and weighted sum steps independently, producing 8 different output vectors (Z0, Z1, ..., Z7) for each input word[cite: 136, 138, 142].

Why multiple heads? Each head can potentially learn to focus on different types of relationships or different aspects of the input sequence (e.g., syntactic dependencies, semantic relationships at different ranges)[cite: 138, 139]. It expands the model's ability to attend to information from different representation subspaces at different positions[cite: 139].

After calculating the output vectors (Z0 to Z7) from all heads, they are concatenated together[cite: 147]. This concatenated vector is then passed through another linear transformation (multiplied by another learned weight matrix, `Wo`) to produce the final output vector for the multi-head attention layer[cite: 147]. This final vector has the desired dimension (e.g., 512) to be passed to the next layer (the feed-forward network)[cite: 143, 144].

The feed-forward network processes each position's vector independently and identically[cite: 141, 145].

### Positional Encoding

Since self-attention processes words in parallel, it doesn't inherently consider the order or position of words in the sequence[cite: 28]. To address this, Transformers add **Positional Encoding** vectors to the input embeddings *before* they enter the encoder/decoder stacks[cite: 148, 151].

Simply adding an index (1, 2, 3...) isn't ideal, especially for long sequences, as the numbers can become very large and unbounded, potentially causing issues during training[cite: 149, 150].

The original paper used **sinusoidal positional encodings**. These are fixed (not learned) vectors calculated using sine and cosine functions of different frequencies based on the position of the word and the dimension within the embedding vector[cite: 153]. The formula ensures that each position gets a unique encoding, and the model can potentially learn to attend to relative positions. The positional encoding vector has the same dimension as the input embedding (e.g., 512), allowing them to be added together[cite: 151, 175]. There are also **learned positional encodings**, where the vectors are learned during training[cite: 154].

The input embedding plus the positional encoding vector forms the final input that goes into the first layer of the encoder/decoder[cite: 155].

### Residual Connections and Layer Normalization

Looking back at the architecture diagram[cite: 7, 155], you'll notice "Add & Norm" layers after both the multi-head attention sub-layer and the feed-forward sub-layer[cite: 156]. This involves two operations:

1.  **Residual Connection (Add):** The input to the sub-layer (e.g., the input to multi-head attention) is added to the output of that sub-layer[cite: 158, 159]. This is also known as a skip connection[cite: 194].
    `SubLayer_Input + SubLayer_Output(SubLayer_Input)` [cite: 159]
    Residual connections help address the vanishing gradient problem in deep networks, allowing gradients to flow more directly through the network during backpropagation[cite: 196]. They also allow the layer to learn modifications to the identity mapping, potentially making training easier[cite: 195].

2.  **Layer Normalization (Norm):** The output of the addition step is then normalized using **Layer Normalization**[cite: 157, 180]. Unlike Batch Normalization, Layer Normalization normalizes the inputs across the features (embedding dimension) for *each* data sample (each word position) independently. It calculates the mean and standard deviation across the embedding dimension for a specific token and normalizes the vector[cite: 161, 162]. This helps stabilize training by keeping the activations within a certain range, irrespective of other samples in the batch[cite: 163]. Layer Normalization includes learnable parameters (gamma and beta) that allow the network to scale and shift the normalized output, potentially restoring representational power if the raw normalized output isn't optimal[cite: 164, 165, 166].

This Add & Norm structure is applied after *each* of the two sub-layers in every encoder and decoder layer[cite: 193].

### Feed-Forward Network

The second sub-layer in each encoder and decoder is a position-wise Feed-Forward Network (FFN)[cite: 48, 54]. This consists of two linear transformations with a ReLU activation in between:

`FFN(x) = max(0, x*W1 + b1)*W2 + b2`

This network is applied independently and identically to each position (each word's vector representation)[cite: 197]. While the self-attention layers handle mixing information across positions, the FFN processes the information *at* each position. It adds representational capacity and depth to the model[cite: 197, 198]. The parameters (W1, b1, W2, b2) are shared across positions within a layer but are different between layers. The inner layer typically has a larger dimension (e.g., 2048 in the original paper) than the input/output dimension (e.g., 512)[cite: 183]. It increases model parameters, potentially helping generalization[cite: 199, 200].

### Encoder Architecture Summary

So, a single encoder layer works as follows (based on the research paper's parameters):
1.  Input: Sequence of vectors (dimension 512, input embedding + positional encoding)[cite: 171, 175].
2.  Multi-Head Self-Attention: Uses 8 attention heads, with Q, K, V dimensions of 64 each. Scaling factor is `sqrt(64) = 8`[cite: 176, 177, 186].
3.  Add & Norm: Residual connection adds input (from step 1) to the output of attention, followed by Layer Normalization[cite: 178, 179, 181, 182].
4.  Feed-Forward Network: Two linear layers with ReLU, inner dimension often larger (e.g., 2048)[cite: 182, 183].
5.  Add & Norm: Residual connection adds input (from step 3) to the output of FFN, followed by Layer Normalization[cite: 182].
6.  Output: Sequence of vectors (dimension 512) passed to the next encoder layer or the decoder stack[cite: 184].

This stack of N=6 encoders processes the input sequence[cite: 169, 170]. The complexity (multiple layers, high dimensions) is needed for complex sequence-to-sequence tasks like machine translation[cite: 187, 188].

### Decoder Architecture

The decoder's role is to generate the output sequence, word by word[cite: 203]. It shares similarities with the encoder but has key differences:

1.  **Masked Multi-Head Self-Attention:** The first sub-layer is a self-attention mechanism, but it's "masked"[cite: 207, 208, 211]. This mask ensures that when predicting the word at position `i`, the decoder can only attend to positions less than or equal to `i` in the *output* sequence. It prevents the decoder from "cheating" by looking ahead at future words it's supposed to predict[cite: 228, 229]. This is crucial during training when the entire target sequence is available. The masking is typically implemented by setting the attention scores corresponding to future positions to a very large negative number (like -infinity) before the softmax step, making their attention weights effectively zero[cite: 245].

2.  **Encoder-Decoder Attention:** The second sub-layer is a multi-head attention mechanism, but it's different from the self-attention in the encoder[cite: 209, 210, 211]. Here, the **Queries (Q)** come from the output of the previous decoder sub-layer (the masked self-attention layer), while the **Keys (K) and Values (V)** come from the **output of the encoder stack**[cite: 252, 253, 254, 255]. This allows the decoder to attend to relevant parts of the *input* sequence while generating each output word[cite: 257].

3.  **Feed-Forward Network:** The third sub-layer is the same position-wise FFN as in the encoder[cite: 210].

Like the encoder, each sub-layer in the decoder is also followed by an Add & Norm step (residual connection + layer normalization)[cite: 260].

### Decoder Operation: Training vs. Inference

**Training:**
During training, the decoder receives the complete target output sequence as input, but shifted one position to the right and typically prepended with a start-of-sequence token (`<SOS>`)[cite: 216, 217]. Padding might be used to make sequences of equal length[cite: 217, 218]. The masked self-attention ensures that the prediction for position `i` only depends on the known outputs at positions `< i`[cite: 220]. The encoder-decoder attention uses K and V from the encoder output and Q from the decoder's masked self-attention output[cite: 255, 256]. The final output of the decoder stack goes through a linear layer and softmax to predict the probability distribution over the vocabulary for the next word[cite: 280, 281]. The loss (e.g., cross-entropy) is calculated between the predicted distribution and the actual target word at each position[cite: 301, 302].

Example: Translating "Je suis étudiant" (I am student) to "I am a student <EOS>".
* Input to Encoder: "Je", "suis", "étudiant"
* Input to Decoder (shifted right): `<SOS>`, "I", "am", "a", "student"
* Target Output for Loss: "I", "am", "a", "student", `<EOS>`

At step 1, decoder sees `<SOS>`, attends to encoder output, predicts "I".
At step 2, decoder sees `<SOS>`, "I", attends to encoder output (masked self-attention only sees `<SOS>`, "I"), predicts "am".
...and so on.

**Inference (Generation):**
During inference, we generate the output one word at a time[cite: 204].
1.  Feed the input sequence to the encoder to get the K and V vectors.
2.  Feed a start token (`<SOS>`) as the initial input to the decoder.
3.  Run the decoder stack. The final linear layer and softmax predict the probability distribution for the first output word. Choose the word with the highest probability (or sample from the distribution).
4.  Feed the generated word as input to the decoder at the next time step.
5.  Repeat step 3 and 4, feeding the previously generated word back into the decoder, until an end-of-sequence token (`<EOS>`) is generated or a maximum length is reached[cite: 270, 271, 272, 273].

### Final Linear Layer and Softmax

After the final decoder layer in the stack processes the sequence, it outputs a sequence of vectors[cite: 283]. To get the final output probabilities for words, this vector sequence is passed through:

1.  **Linear Layer:** A final, simple fully connected neural network layer[cite: 285]. It projects the high-dimensional vector output from the decoder stack (e.g., 512 dimensions) into a much larger vector called the **logit vector**[cite: 285]. The size of this logit vector is equal to the size of the vocabulary (e.g., 10,000 unique words)[cite: 287, 288]. Each element in the logit vector corresponds to a score for a unique word in the vocabulary[cite: 289, 290].

2.  **Softmax Layer:** The logit vector is then passed through a softmax layer[cite: 291]. This converts the raw scores (logits) into probabilities, ensuring they are all non-negative and sum to 1[cite: 294]. Each probability represents the model's confidence that the corresponding word is the correct next word in the output sequence[cite: 293, 294]. The word associated with the highest probability is typically chosen as the output for that time step during inference[cite: 295].

### Training Process Recap and Loss Function

During training, we compare the predicted probability distribution (from the final softmax layer) with the actual target word (usually represented as a one-hot encoded vector across the vocabulary)[cite: 298, 299, 300]. A **loss function**, typically cross-entropy loss, measures the difference between the predicted and target distributions[cite: 301, 302]. This loss is then backpropagated through the entire network (decoder and encoder) to update all the learned weight matrices (`Wq`, `Wk`, `Wv`, `Wo`, FFN weights, linear layer weights, embedding weights if learned) using gradient descent or a variant like Adam[cite: 307, 308]. The goal is to minimize this loss, making the model's predictions closer to the actual target sequences over many training epochs[cite: 305, 306, 308].
