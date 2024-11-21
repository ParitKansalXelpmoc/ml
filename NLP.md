
# NLP

## Content
- [Text Preprocessing Steps](#text-preprocessing-steps)
- [Text Representation & Text Embeddings](#text-representation--text-embeddings)
    - [One-Hot Encoding](#one-hot-encoding)
    - [Bag of Words (BoW)](#bag-of-wordbow)
    - [N-Gram](#n-gram)
    - [TF-IDF (Term Frequency-Inverse Document Frequency)](#tf-idf-term-frequency-inverse-document-frequency)
    - [Word2Vec](#word2vec)

## Text Preprocessing Steps

- **Lowercasing**  
   Convert all text to lowercase for uniformity.
- **Remove HTML Tags**  
   Eliminate HTML tags like `<div>` or `<p>` to retain only the plain text.
- **Remove URLs**  
   Strip out any web links from the text.
- **Remove Punctuation**  
   Remove punctuation marks to simplify the text.
- **Chat Word Treatment**  
   Replace common chat abbreviations or slang (e.g., 'u' → 'you', 'r' → 'are').
- **Spelling Correction**  
   Correct misspelled words to their standard forms.
- **Removing Stop Words**  
   Exclude common words like "and", "is", and "the" that do not contribute much meaning.
- **Handling Emojis**  
   Remove or replace emojis with their textual description.
- **Tokenization**  
   - **Word Tokenization**: Break the text into individual words.  
   - **Sentence Tokenization**: Divide the text into sentences.
- **Stemming**  
    Reduce words to their root forms, even if the resulting word lacks meaning (e.g., "running" → "run").
- **Lemmatization**  
    Reduce words to their meaningful base forms (e.g., "better" → "good").

---
---

## Text Representation / Text Embeddings

#### **Common Terms**:

- **Corpus**: A collection of text data used for analysis or training models.
- **Vocabulary**: The unique set of words or tokens in the corpus.
- **Document**: A single piece of text (e.g., a sentence, paragraph, or article) in the corpus.
- **Word**: An individual token from the vocabulary.

---

### One Hot Encoding
- **Steps**:
    - Identify the **Vocabulary** from the corpus.
    - Represent each word using a sparse vector based on the vocabulary, with a single "1" indicating the word's presence.

- **Example**:

    | Document | Content                      |
    |----------|------------------------------|
    | D1       | people watch campusx         |
    | D2       | campusx watch campusx        |
    | D3       | people write comment         |
    | D4       | campusx write comment        |

    Vocabulary = [people, watch, campusx, write, comment]

    | Document | Content                      | Vector                               |
    |----------|------------------------------|--------------------------------------|
    | D1       | people watch campusx         | \[ [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0] \] |
    | D2       | campusx watch campusx        | \[ [0,0,1,0,0], [0,1,0,0,0], [0,0,1,0,0] \] |
    | D3       | people write comment         | \[ [1,0,0,0,0], [0,0,0,1,0], [0,0,0,0,1] \] |
    | D4       | campusx write comment        | \[ [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1] \] |

- Pocs:
    - Intutive
    - Easy Implementation
- Cons:
    - Sparsity
    - No Fixed Size
    - Out Of Vocabulary
    - No campturing of semantic

---

### Bag Of Word(BOW)

- **Steps**:
  - Identify the **Vocabulary**.
  - Represent each document as a fixed-size vector where each unit is the count of a word in the document.
- **Example**:

    | Document | Content                      |
    |----------|------------------------------|
    | D1       | people watch campusx         |
    | D2       | campusx watch campusx        |
    | D3       | people write comment         |
    | D4       | campusx write comment        |

    Vocabulary = [people, watch, campusx, write, comment]

    | Document | Content                |Vector     |Binary Vector|
    |----------|------------------------|-----------|-------------|
    | D1       | people watch campusx   |[1,1,1,0,0]|[1,1,1,0,0]  |
    | D2       | campusx watch campusx  |[0,1,2,0,0]|[0,1,1,0,0]  |
    | D3       | people write comment   |[1,0,0,1,1]|[1,0,0,1,1]  |
    | D4       | campusx write comment  |[0,0,1,1,1]|[0,0,1,1,1]  |

- Pocs:
    - Intutive
    - Easy Implementation
    - Fixed Size
- Cons:
    - Sparsity
    - Out Of Vocabulary
    - Ordering Get Changed
    - No campturing of semantic

---

### N Gram

- **Steps**:
  - Build a vocabulary using **N-word combinations**.
  - Represent each document as a vector where each unit indicates the count of N-grams.

- **Example**:

    | Document | Content                      |
    |----------|------------------------------|
    | D1       | people watch campusx         |
    | D2       | campusx watch campusx        |
    | D3       | people write comment         |
    | D4       | campusx write comment        |

   
    Vocabulary = [people watch, watch campusx, campusx watch, watch campusx, people write, write comment, campusx write]  

    ### Updated Representation:  

    | Document | Content                | Vector         |
    |----------|------------------------|---------------|
    | D1       | people watch campusx   | \[1, 1, 0, 0, 0, 0, 0\] |
    | D2       | campusx watch campusx  | \[0, 0, 1, 1, 0, 0, 0\] |
    | D3       | people write comment   | \[0, 0, 0, 0, 1, 1, 0\] |
    | D4       | campusx write comment  | \[0, 0, 0, 0, 0, 1, 1\] |


- Pocs:
    - Able of campturing of semantic
    - Intutive
    - Easy Implementation
    - Fixed Size
- Cons:
    - Dimension Increses
    - Out Of Vocabulary

---

### TF-IDF (Term Frequency-Inverse Document Frequency)

- **Steps:**
    - Apply Bag of Words:
    - Calculate Term Frequency(Tf): $TF(d, t) = \frac{\text{Number of occurrences of term } t \text{ in document } d}{\text{Total number of terms in document } d}$
    - Calculate Inverse Document Frequency (IDF): $IDF(t) = \ln\left(\frac{\text{Total number of documents in the corpus}}{\text{Number of documents containing term } t}\right)$
    - Compute TF-IDF Weight: $W(d, t) = TF(d, t) \times IDF(t)$
- **Example**:

    | Document | Content                      |
    |----------|------------------------------|
    | D1       | people watch campusx         |
    | D2       | campusx watch campusx        |
    | D3       | people write comment         |
    | D4       | campusx write comment        |

    Vocabulary = [people, watch, campusx, write, comment]

    **BOW**->
    | Document | Content                |Vector     |
    |----------|------------------------|-----------|
    | D1       | people watch campusx   |[1,1,1,0,0]|
    | D2       | campusx watch campusx  |[0,1,2,0,0]|
    | D3       | people write comment   |[1,0,0,1,1]|
    | D4       | campusx write comment  |[0,0,1,1,1]|

    **Tf**->
    | Document | people | watch | campusx | write | comment |
    |----------|--------|-------|---------|-------|---------|
    | D1       | 0.333  | 0.333 | 0.333   | 0.000 | 0.000   |
    | D2       | 0.000  | 0.333 | 0.667   | 0.000 | 0.000   |
    | D3       | 0.333  | 0.000 | 0.000   | 0.333 | 0.333   |
    | D4       | 0.000  | 0.000 | 0.333   | 0.333 | 0.333   |

    **IDF**->
    | Term      | people | watch | campusx | write | comment |
    |-----------|--------|-------|---------|-------|---------|
    | IDF Value | 0.693  | 0.693 | 0.287   | 0.693 | 0.693   |

    **Final TF-IDF (W) Matrix**->
    
    | Document | people | watch | campusx | write | comment |
    |----------|--------|-------|---------|-------|---------|
    | D1       | 0.231  | 0.231 | 0.096   | 0.000 | 0.000   |
    | D2       | 0.000  | 0.231 | 0.191   | 0.000 | 0.000   |
    | D3       | 0.231  | 0.000 | 0.000   | 0.231 | 0.231   |
    | D4       | 0.000  | 0.000 | 0.096   | 0.231 | 0.231   |

    **Final TF-IDF matrix**->
    
    | Document | people | watch | campusx | write | comment |
    |----------|--------|-------|---------|-------|---------|
    | D1       | 0.231  | 0.231 | 0.096   | 0.000 | 0.000   |
    | D2       | 0.000  | 0.231 | 0.382   | 0.000 | 0.000   |
    | D3       | 0.231  | 0.000 | 0.000   | 0.231 | 0.231   |
    | D4       | 0.000  | 0.000 | 0.096   | 0.231 | 0.231   |

- Procs
    - Information Retrival System
- Cons:
    - Sparsity
    - Out Of Vocabulary
    - Ordering Get Changed
    - No campturing of semantic

---

### Word2Vec

#### CBOW

- Make a window of odd size. Let the window size be 3.
- The context words (word1, word3) are used to predict the target word (word2) ->  word1 ....?....  word3
- Convert the words to one-hot encoding vectors.
- feed it to neuron network given below 

![](https://github.com/ParitKansal/photos/blob/main/CBOW.png)


#### Skip Gram
- Make a window of odd size. Let the window size be 3.
- The target word (word2) is used to predict the context words (word1, word3). -> ....?.... word2 ....?....
- Convert the words to one-hot encoding vectors.
- Feed it to neuron network given below

![](https://github.com/ParitKansal/photos/blob/main/SkipGram.png)

---
