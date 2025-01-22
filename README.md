## NLP (Natural Language Processing) and Named Entity Recognition (NER)

NLP is the process by which we get a computer system to understand, parse, and extract human language, often from raw text. NLP includes various tasks such as:

- Speech tagging
- Syntactic parsing
- Text categorization (also known as text classification)
- Coreference resolution
- Machine translation

NLP is closely related to computational linguistics.

### spaCy

- **spaCy** is a free, open-source library for advanced Natural Language Processing (NLP) in Python.
- It was designed by Explosion AI, with the goal of working at scale with AI systems—processing large quantities of documents efficiently, effectively, and accurately.
- It can process hundreds of thousands of documents with ease, especially when using rules-based pipelines.

### Common NLP Tasks

| Name                           | Description                                                                                      |
|--------------------------------|--------------------------------------------------------------------------------------------------|
| **Tokenization**               | Segmenting text into words, punctuation marks, etc.                                               |
| **Part-of-Speech (POS) Tagging**| Assigning word types to tokens, such as verb, noun, etc.                                          |
| **Dependency Parsing**         | Assigning syntactic dependency labels that describe relationships between tokens (e.g., subject, object). |
| **Lemmatization**              | Assigning the base forms of words (e.g., “was” → “be” or “rats” → “rat”).                        |
| **Sentence Boundary Detection (SBD)** | Finding and segmenting individual sentences.                                               |
| **Named Entity Recognition (NER)**  | Labeling named “real-world” objects, such as persons, companies, or locations.                  |
| **Entity Linking (EL)**        | Disambiguating textual entities to unique identifiers in a knowledge base.                        |
| **Similarity**                 | Comparing words, text spans, and documents to measure how similar they are.                      |
| **Text Classification**        | Assigning categories or labels to whole documents or parts of documents.                        |
| **Rule-based Matching**        | Finding sequences of tokens based on their text and linguistic annotations, similar to regular expressions. |
| **Training**                   | Updating and improving a statistical model’s predictions.                                        |
| **Serialization**              | Saving objects to files or byte strings.                                                         |

---

### Containers in spaCy

In spaCy, containers are objects that hold large quantities of data about a text. There are several types of containers in spaCy, such as:

- **Doc**
- **DocBin**
- **Example**
- **Language**
- **Lexeme**
- **Span**
- **SpanGroup**
- **Token**

#### Doc Object
The `Doc` object in spaCy contains a lot of information about the text passed to the spaCy pipeline. It includes attributes such as:
- **Sentences**: You can access individual sentences in a `Doc` object.
- **Tokens**: Tokens are the individual units (words or punctuation) within the document.

#### Spans
Spans in spaCy represent sequences of tokens. They can be:
- A single token (e.g., "Berlin").
- A group of tokens (e.g., "Martin Luther King").

The `Doc` object has a different length than the original text. The `Doc` counts individual tokens, while the text object doesn't.

---

### Sentence Boundary Detection

To extract sentences from a `Doc` object, use `for sent in doc.sents`. To convert them into a list, use `list(doc.sents)`.

---

### Linguistic Features in spaCy

- **Part-of-Speech Tagging**: spaCy uses a trained pipeline and statistical models to parse and tag a given document. These models make predictions using binary data, which helps reduce memory usage and improves efficiency by encoding strings into hash values.
  
- **Morphology**:
  - **Inflectional Morphology**: Modifies the root form of a word by adding prefixes or suffixes that specify grammatical function.
  - **Rule-based Morphology**: Uses token text and part-of-speech tags to produce morphological features (e.g., `token.tag`).

- **Lemmatization**: 
  - spaCy provides two pipeline components for lemmatization:
    1. The **Lemmatizer** component, which uses lookup and rule-based methods.
    2. The **EditTreeLemmatizer**, which provides a trainable lemmatizer.

---

### Dependency Parsing

- **Text**: The original noun chunk text.
- **Root Text**: The original text of the word that connects the noun chunk to the rest of the sentence.
- **Root Dep**: The dependency relation connecting the root to its head.
- **Root Head Text**: The text of the root token’s head.

#### Noun Chunks

- **Text**: The original token text.
- **Dep**: The syntactic relation connecting the child to the head.
- **Head Text**: The original text of the token head.
- **Head POS**: The part-of-speech tag of the token head.
- **Children**: The immediate syntactic dependents of the token.

- **Subtree**: The subtree includes all tokens dominated by a given token, including the token itself.

---

### Named Entity Recognition (NER)

A named entity is a "real-world object" assigned a name, such as a person, country, product, or book title. spaCy can recognize and classify various types of named entities in a document through a prediction model.

# Attribute and Matching Techniques in spaCy

## **Attributes**
| Attribute  | Description                               |
|------------|-------------------------------------------|
| **ORTH**   | The exact verbatim text of a token.       |
| **ENT_TYPE** | The token’s entity label (e.g., PERSON, ORG). |

---

## **Regex Matching**
- The `REGEX` operator allows defining rules for any attribute string value, including custom attributes.
- Must be applied to an attribute like `TEXT`, `LOWER`, or `TAG`.

---

## **Fuzzy Matching**
  - Fuzzy matching allows matching tokens with alternate spellings, typos, or similar variations.
  - Handles misspellings and minor errors in text.
- **ex:** 
  - Searching for names, places, or phrases where errors are common.
  - Matching text with small variations without specifying every possible variant.

---

## **Matching Techniques**

### **Phrase Patterns**
- Match exact words or sequences of words (phrases).
- You know the exact words or phrases you want to match.
- You don’t care about attributes like part-of-speech tags or lemmas.
- `PhraseMatcher`

---

### **Token Patterns**
-  Match sequences of tokens based on their attributes (e.g., part-of-speech, lowercase forms, etc.).
- You need more flexibility, such as matching adjectives followed by nouns.
- You want to match based on linguistic features like POS tags, dependency relations, or lemmas.
-  `Matcher`
## Bag of Words

- Initial methods for creating word vectors in a pipeline take all words in a corpus and convert them into a single, unique number. 
- This is known as a **bag of words**.
v2:
# Stemming in NLP

**Stemming** is the process of reducing a word to its word stem by removing suffixes and prefixes, or by mapping it to the root form of the word (which can also be called a "lemma"). Stemming is an important concept in **Natural Language Understanding (NLU)** and **Natural Language Processing (NLP)**.

### Example: 
- Words like **eating**, **eats**, and **eaten** are reduced to the root form **eat**.

### Why is Stemming Important?
Stemming is important because it helps in handling different forms of the same word, making it easier to classify and analyze text, especially for tasks like sentiment analysis and text classification.

### Challenges in Stemming:
- Sometimes stemming might not return the correct meaning of the word. For instance, the word "history" might be stemmed to "histori", which is not a valid root form of the word.

## Sentiment Classification Problem:
In the context of a **classification problem** like determining whether product reviews are **positive** or **negative**, stemming helps in reducing variations of words (like "eating", "eats", "eaten") to their common stem ("eat").

---

## Stopwords in NLTK vs. spaCy

### NLTK:
In **NLTK**, you use `stopwords.words("english")` to directly access the list of stopwords.

### spaCy:
In **spaCy**, you access stopwords using the language model's `Defaults.stop_words`.
# Preprocessing in NLP

Preprocessing is essential to clean the data and ensure that it's ready for conversion into vectors in an efficient way. One of the key preprocessing techniques is **Lemmatization**, which addresses issues like variations in word forms by returning the word to its base form.

## Lemmatization

Lemmatization solves many language-related issues because it uses a dictionary of words and their corresponding root forms. For example, if you input the word **"goes"**, lemmatization will return **"go"**, which is the correct grammatical form of the word.

### Lemmatizer:
Lemmatization techniques are similar to **stemming**. However, the difference is that lemmatization outputs a **valid word** rather than a stem. The word obtained after lemmatization is called a **lemma** (e.g., "goes" becomes the root word **"go"**).

Unlike stemming, lemmatization guarantees that the output is a valid word with the same meaning.

---

## Word Embedding

**Word Embedding** is a technique in **Natural Language Processing (NLP)** used to represent words for text analysis. This representation is typically in the form of a real-valued vector that encodes the meaning of a word. In this space, words that are closer to each other in the vector space are expected to have similar meanings.

---

## Word2Vec

**Word2Vec** is a technique used in **Natural Language Understanding (NLU)**. The **Word2Vec algorithm** uses a neural network model to learn word associations from a large corpus of text. Once the model is trained, it can:
- Detect synonymous words
- Suggest additional words for partial sentences

As the name implies, Word2Vec represents each distinct word with a list of numbers called a **vector**.

# Transformers

Transformers are powerful models used in NLP and other domains. They consist of **encoders** and **decoders** that work together for tasks like translation, summarization, and text generation.

## Encoders and Decoders

- **Encoder**: Takes an input sentence (e.g., *"je suis étudiant"* in French) and converts it into a hidden representation that captures the sentence's meaning.  
- **Decoder**: Transforms the hidden representation into the desired output language (e.g., *"I am a student"* in English).

## How Transformers Work

1. **Tokenization**:  
   The input sentence is tokenized and converted into vectors.

2. **Encoder Processing**:  
   These vectors pass through multiple layers of the encoder, which include:
   - **Feedforward Neural Networks**  
   - **Self-Attention Mechanisms**

3. **Decoder Processing**:  
   - The output of the encoder is passed to the decoder.  
   - The decoder uses:
     - **Feedforward Neural Networks**  
     - **Cross-Attention** (to focus on the encoder's output while generating the target sentence).

4. **Output Generation**:  
   The decoder generates the output sentence token by token.

---



