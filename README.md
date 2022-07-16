# Bangla-Fake-News-Detection-System #
##### (GitHub upload not yet completed................)

## Summary ##
The Spread of fake news is a global issue. In an age where the internet is commonly used as the primary source of information, news consumers are more likely than ever to come across and share fake news. Consumers all around the world read, watch, or listen to the news every day for updates on everything from their favorite celebrity to their favored political candidate, and they typically assume that what they discover is accurate and trustworthy. It is a very difficult and time consuming approach for detecting fake news by a human himself. A machine learning model can help people to detect fake news within a very short time. As a result, fake news detection has become a very popular topic for the researchers. In this work, two methodologies have been proposed to detect the Bangla fake news, which are hashing and hashing-autoencoder. Hashing transforms a text into a set of indexes that has a fixed size. It puts the features in a smaller dimensional space. It reduces the original high dimensional space by mapping the features to hash keys. Autoencoder is a neural network to clean up noisy or sparse data. The input vector is rebuilt by moving from a high-dimensional space to a lower-dimensional space and then back to a higher-dimensional space. The new methodologies have been compared to traditional methodologies using natural language based neural networks such as LSTM & GRU and hybrid networks CNN-LSTM & CNN-GRU. Some machine learning models also have been used such as Logistic Regression, Decision Tree & Passive Aggressive Classifier with random parameter tuning and ensemble based methods such as Voting & Boosting. The Logistic Regression, the Light Gradient Boosted Machine and the CNN-LSTM hybrid network perform better by giving the accuracy of above 90% for detecting the fake news.

## Contribution ##
The main contribution of this work is in the feature extraction part of the methodology. The main contribution of this thesis/project are:

`- Using the hashing trick for the feature extraction from the text corpus. In general, TF-IDF, bag of words, word embedding, etc. techniques are being used widely for feature extraction. For both ML and DL algorithms hashing is used. For the large data set, it has improved the performance of the model significantly.`

`- Using the hashing-autoencoder technique for ML algorithms. With the help of a hashing trick we used an autoencoder technique for regeneration of the input for reducing the sparsity of the input vectors. It has improved the model’s performance in a more balanced way than only using the hashing trick.`

`- Finding the best learning model for detecting the fake news from the Bangla news corpus using the proposed methods.`

## Basic Framework Overview ##
![Basic Framework](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Framework-Overview.drawio.png)

Figure shows the basic overview of the proposed system. Here we use both ML and DL models with feature extraction techniques commonly used TF-IDF technique and word embedding, rarely used `hashing trick` technique and also a `hashing-autoencoder model` for reconstructing the input vector for various ML models which are Logistic Regression, Passive Aggressive Classifier, Decision Tree algorithm and ensemble learning like Voting Classifiers, Boosting algorithms etc. In DL models, we have used both LSTM, GRU and also hybrid networks with CNN-LSTM, CNN-GRU. Among them the hashing trick for feature extraction is totally new for Bangla Fake News Detection, and hashing autoencoder for both feature extraction & input regeneration is totally new for fake news detection of any language corpus. These two types of proposed system have been outperformed over other traditional approaches with the performance of detecting fake news.

### Methodology 1.1 - Hashing Trick for Machine Leanring
![Hashing for ML](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Hashing-ML-Methodo.drawio.png))

Figure shows the methodology of the Hashing technique for ML algorithms in Bangla fake news detection. 

- Hashing trick also known as feature hashing. Feature hashing is a sophisticated method of modeling data sets with a lot of component and character information. It takes up less memory and doesn’t require much pre-processing [12]. It is memory efficient. In TF-IDF or Bag of Words, tokens are saved as strings, and in this method the tokens are encoded as integer indices. Hashing converts a string to a set of indexes in a hashing space with a defined size that means the features into a lower-dimensional space, i.e. mapping features to hash keys, reduces the original high-dimensional space. A single key can be assigned to many features. The collided features then take the other vacant positions in the other vacant positions. Using a second hash function to signal the sign (+/-) to apply to the value being changed within the vector is another strategy that is occasionally used. If two words have the same vector space index, one of them may have a result of +1 (or +n, where n is the number of times the word appears in the relevant document), whereas the other may have a result of -1. (or -n where n is the number of occurrences of the word within the corresponding document). This prevents the mistake from being compounded by the collision by having the two word occurrences cancel each other out. It is the processes that run in parallel.
- The hash function that is used is known as Mumurhash3 which is a multiply and rotation hash. Here two multiplications and two rotations take place. It takes a reference to a byte stream, byte stream length and a random state or seed. The equation shows the mathematical formulas of Mumurhash3: Here: b is the block key, a is the hash states block and n1, n2, c1 are constant. In the `sklearn` module it is a default hash function in `Hashing Vectorizer`.

```

b = b × n1
b = rotation(b, n1)
b = b × n2
a = a ⊕ b
a = rotation(a, n2)
a = a × c1 + n1

'''
