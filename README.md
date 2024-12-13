![GitHub last commit](https://img.shields.io/github/last-commit/nafiul-araf/Bangla-Fake-News-Detection-System)
![GitHub code size](https://img.shields.io/github/languages/code-size/nafiul-araf/Bangla-Fake-News-Detection-System)
![GitHub repo size](https://img.shields.io/github/repo-size/nafiul-araf/Bangla-Fake-News-Detection-System)
![GitHub all releases](https://img.shields.io/github/downloads/nafiul-araf/Bangla-Fake-News-Detection-System/total)
![GitHub language count](https://img.shields.io/github/languages/count/nafiul-araf/Bangla-Fake-News-Detection-System)
![GitHub top language](https://img.shields.io/github/languages/top/nafiul-araf/Bangla-Fake-News-Detection-System?color=yellow)

# A Hashing and AutoEncoder based Bangla Fake News Detection System 

> The demo app is deployed using [Streamlit](https://golden.com/wiki/Streamlit-5KMKYAB)

```
Input -> A news article text
Output -> Fake or Real with a confidence ranges from 0 to 1
```

Link:-> 👇
[![Interface](https://github.com/nafiul-araf/BFND-System/blob/main/Interface.PNG)](https://bfnd-system-3ah3m5kckw6q3fuzc3vdh2.streamlit.app/)

> Sign in/up to github account to see deployment environment hisory

### -------------------->>>>>>Running the demo API on local host<<<<<<--------------------- ###

```
open the git bash and type the command: git clone https://github.com/nafiul-araf/BFND-System.git
```

``` 
unzip into the folder
```

```
Go to the folder 'APP' and in the command prompt type: streamlit run App.py
```

> [Demo Api 1](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/demo1.PNG)

> [Demo Api 2](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/demo2.PNG)

> [Demo Api 3](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/demo3.PNG)

> [Demo Api 4](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/demo4.PNG)

> [Demo Api 5](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/demo5.PNG)

## Summary ##
The Spread of fake news is a global issue. In an age where the internet is commonly used as the primary source of information, news consumers are more likely than ever to come across and share fake news. Consumers all around the world read, watch, or listen to the news every day for updates on everything from their favorite celebrity to their favored political candidate, and they typically assume that what they discover is accurate and trustworthy. It is a very difficult and time consuming approach for detecting fake news by a human himself. A machine learning model can help people to detect fake news within a very short time. As a result, fake news detection has become a very popular topic for the researchers. In this work, two methodologies have been proposed to detect the Bangla fake news, which are hashing and hashing-autoencoder. Hashing transforms a text into a set of indexes that has a fixed size. It puts the features in a smaller dimensional space. It reduces the original high dimensional space by mapping the features to hash keys. Autoencoder is a neural network to clean up noisy or sparse data. The input vector is rebuilt by moving from a high-dimensional space to a lower-dimensional space and then back to a higher-dimensional space. The new methodologies have been compared to traditional methodologies using natural language based neural networks such as LSTM & GRU and hybrid networks CNN-LSTM & CNN-GRU. Some machine learning models also have been used such as Logistic Regression, Decision Tree & Passive Aggressive Classifier with random parameter tuning and ensemble based methods such as Voting & Boosting. The Logistic Regression, the Light Gradient Boosted Machine and the CNN-LSTM hybrid network perform better by giving the accuracy of above 90% for detecting the fake news.

```Declaration of Originality``` [^1]

[^1]: I certify that, to the best of my knowledge, my thesis does not infringe upon anyone’s copyright nor violate any proprietary rights and that any ideas, techniques, quotations, or any other material from the work of other people included in my thesis, published or otherwise, are fully acknowledged in accordance with the standard referencing practices.   
![sign](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/signature.png)

## Contribution ##
The main contribution of this work is in the `feature extraction` part of the methodology. The main contributions of this thesis/project are:

```
Using the hashing trick for the feature extraction from the text corpus. In general, TF-IDF, bag of words, word embedding, etc. techniques are being used widely for feature extraction. For both ML and DL algorithms hashing is used. For the large data set, it has improved the performance of the model significantly
```

```
Using the hashing-autoencoder technique for ML algorithms. With the help of a hashing trick we used an autoencoder technique for regeneration of the input for reducing the sparsity of the input vectors. It has improved the model’s performance in a more balanced way than only using the hashing trick
```

```
Finding the best learning model for detecting the fake news from the Bangla news corpus using the proposed methods
```

## Basic Framework Overview ##
![Basic Framework](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Framework-Overview.drawio.png)

Figure shows the basic overview of the proposed system. Here we use both ML and DL models with feature extraction techniques commonly used TF-IDF technique and word embedding, rarely used `hashing trick` technique and also a `hashing-autoencoder model` for reconstructing the input vector for various ML models which are Logistic Regression, Passive Aggressive Classifier, Decision Tree algorithm and ensemble learning like Voting Classifiers, Boosting algorithms etc. In DL models, we have used both LSTM, GRU and also hybrid networks with CNN-LSTM, CNN-GRU. Among them the hashing trick for feature extraction is totally new for Bangla Fake News Detection, and hashing autoencoder for both feature extraction & input regeneration is totally new for fake news detection of any language corpus. These two types of proposed system have been outperformed over other traditional approaches with the performance of detecting fake news.

### Methodology 1.1 - Hashing Trick for Machine Leanring
![Hashing for ML](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Hashing-ML-Methodo.drawio.png)

Figure shows the methodology of the Hashing technique for ML algorithms in Bangla fake news detection. 

- Hashing trick also known as feature hashing. Feature hashing is a sophisticated method of modeling data sets with a lot of component and character information. It takes up less memory and doesn’t require much pre-processing [12]. It is memory efficient. In TF-IDF or Bag of Words, tokens are saved as strings, and in this method the tokens are encoded as integer indices. Hashing converts a string to a set of indexes in a hashing space with a defined size that means the features into a lower-dimensional space, i.e. mapping features to hash keys, reduces the original high-dimensional space. A single key can be assigned to many features. The collided features then take the other vacant positions in the other vacant positions. Using a second hash function to signal the sign (+/-) to apply to the value being changed within the vector is another strategy that is occasionally used. If two words have the same vector space index, one of them may have a result of +1 (or +n, where n is the number of times the word appears in the relevant document), whereas the other may have a result of -1. (or -n where n is the number of occurrences of the word within the corresponding document). This prevents the mistake from being compounded by the collision by having the two word occurrences cancel each other out. It is the processes that run in parallel.
- The hash function that is used is known as Mumurhash3 which is a multiply and rotation hash. Here two multiplications and two rotations take place. It takes a reference to a byte stream, byte stream length and a random state or seed [13]. The equation shows the mathematical formulas of Mumurhash3 [14]: Here: b is the block key, a is the hash states block and n1, n2, c1 are constant. In the `sklearn` module it is a default hash function in `Hashing Vectorizer`.

```
b = b × n1
b = rotation(b, n1)
b = b × n2
a = a ⊕ b
a = rotation(a, n2)
a = a × c1 + n1
```

### Methodology 1.2 - Hashing Trick for Machine Leanring
![Hashing for DL](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Hashing-DL-Methodo.drawio.png)

The Figure shows the methodology of the Hashing technique for DL algorithms in Bangla fake news detection.

- In DL, MD5 hash function is used. It is a stable hash function. It turns data into a 32 digits hexadecimal numbers. To generate a hash, the MD5 hashing technique employs a complicated mathematical formula. It divides data into blocks of specific sizes and manipulates it many times. The algorithm adds a Methodology 24 unique value to the computation and turns the result into a small signature or hash while this is going on. In `keras` framework it is available in default as `hashing_trick`.

### Methodology 2 - Hashing-Autoencoder for Machine Leanring
![Hash-Autoen for ML](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Hashing-Autoencoder-Methodo.drawio.png)

The figure shows the methodology of the Hashing-Autoencoder technique only for ML algorithms in Bangla fake news detection.

- After using Hashing for feature extraction, we also used an Autoencoder for reconstructing the feature vectors. It is a neural network based Encoder and Decoder basically used for removing the noise from the data, anomaly detection, reconstructing the data, dimensionality reduction etc. [16]. It is an unsupervised learning. They don’t require specific labels to train on since they don’t need them. However, because they build their own labels from the training data, they are self-supervised [17], [18]. An autoencoder consists of an encoder and a decoder. The encoder part reduces the higher dimension to the lower dimension by compressing the input in a latent dimension. The decoder then regenerates the input from the encoder’s latent dimension, which is returned to the lower dimension’s original dimension. As a result, Complex nonlinear functions can be modelled using auto-encoders. From the lower space to higher space, it removes the noise from the data.

```
We used two encoder blocks and two decoder blocks. All the blocks consist of a dense layer, a batch normalization layer and leaky relu activation function. The decoder block 1 & the encoder block 2 are same, and the decoder block 1 and the encoder block 1 are same as the decoder block regenerates the data. The latent layer is also called a bottleneck layer which consist of a dense layer. The Encoder 1 block has (2 x input) number of neurons & decoder block 2 has the same neurons. The Encoder 2 block has same number as input of neurons & decoder block 1 has the same neurons. The latent layer has (input/2) number of neurons. The below figure shows the graphical representation of the autoencoder that I have used.
```

![Autoencode](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Autoencoder.drawio.png)

#### Deep Learning Model's Architecture ####
- LSTM model architechture

![lstm](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/lstm.drawio.png)

- GRU model architechture

![gru](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/gru.drawio.png)

- CNN-LSTM model architechture

![cnn_lstm](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/cnn_lstm.drawio.png)

- CNN-GRU model architechture

![cnn_gru](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/cnn_gru.drawio.png)

## Results and Discussions ##
We have evaluated the framework using 8501 data. The test data has comefrom different distribution of the train & validation data but from same online source [5]. In addition to my proposed methodology for comparing performance, We have also evaluated the traditional feature extraction techniques TF-IDF for ML models and One Hot Encoding for DL models, respectively. After feature extracting from the test data we have used the trained models for detecting the fake news. Since the authors of this article shared the dataset with the benchmark techniques to identify Bangla fake news, we also compare the performance of our suggested methodology with the paper [5] in the final discussion.

[Result Analysis pdf](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Result%20Analysis.pdf)

[Final Discussions pdf](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/Final%20Discussions.pdf)

**- Hashing Trick for Machine Learning Models**

> Performance comparison with the proposed system

![h_p_ml](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/hashing_performance.png)

**- Hashing Trick for Deep Learning Models**

> Performance comparison with the proposed system

![h_p_dl](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/dl_hashing_performance.png)

**- Hashing-Autoencoder for Machine Learning Models**

> Performance comparison with the proposed system

![h_a_p_ml](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/hashing_autoencoder_performance.png)

**- Comparsion with Base Paper**

> For Machine Learning Models

![b_p_ml](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/paper%20comparison%20ml.PNG)

> For Deep Learning Models

![b_p_dl](https://github.com/nafiul-araf/Bangla-Fake-News-Detection-System/blob/main/paper%20comparison%20dl.PNG)

## Limitations ##
As the words or features can not be regenerated in the hashing trick, it is not possible to identify the features of real or fake news. But the performance is much better than the TF-IDF or One Hot Encoding. Therefore, we must make an assumption. We noticed that the length and word count of the false news material is significantly lower than that of the authentic news information. This will have an impact on how the input text is cleaned up and vectorized. We thus believed that this may be the cause of the models ability to distinguish between authentic and fraudulent news. But much research should be done in these scenarios.

## Future Work ##
We have to thoroughly and repeatedly test the approaches in order to build the models. We were unable to use any further pre-existing strategies like Word2Vec, GloVe, etc. because of time constraints and hardware limitations. This methodology will thus be used to compare my recommended approaches in subsequent research. A number of embedding strategies will also be used, along with transformer models like the BERT and GPT-3. Similar gaps exist in Bangla data on fake news. If more data from diverse scattered sources can be utilized to train the models, more model improvement is possible. To determine the characteristics of authentic or fraudulent news, more study is required due to the limitations of hashing. 
## References ##
[1] Shu, K., Sliva, A., Wang, S., Tang, J., Liu, H.: Fake news detection on social media: A data mining perspective. ACM SIGKDD explorations newsletter 19(1), 22–36 (2017)

[2] Islam, F., Alam, M.M., Hossain, S.S., Motaleb, A., Yeasmin, S., Hasan, M., Rahman, R.M.: Bengali fake news detection. In: 2020 IEEE 10th International Conference on Intelligent Systems (IS), pp. 281–287 (2020). IEEE

[3] Mugdha, S.B.S., Kuddus, M.B.M.M., Salsabil, L., Anika, A., Marma, P.P., Hossain, Z., Shatabda, S.: A gaussian naive bayesian classifier for fake news detection in bengali. In: Emerging Technologies in Data Mining and Information Security, pp. 283–291. Springer, ??? (2021)

[4] P´erez-Rosas, V., Kleinberg, B., Lefevre, A., Mihalcea, R.: Automatic detection of fake news. arXiv preprint arXiv:1708.07104 (2017)

[5] Hossain, M.Z., Rahman, M.A., Islam, M.S., Kar, S.: Banfakenews: A dataset for detecting fake news in bangla. arXiv preprint arXiv:2004.08789 (2020)

[6] Hussain, M.G., Hasan, M.R., Rahman, M., Protim, J., Al Hasan, S.: Detection of bangla fake news using mnb and svm classifier. In: 2020 International Conference on Computing, Electronics & Communications Engineering (iCCECE), pp. 81–85 (2020). IEEE

[7] Kaur, S., Kumar, P., Kumaraguru, P.: Automating fake news detection system using multi-level voting model. Soft Computing 24(12), 9049–9069 (2020)

[8] Da Silva, N.F., Hruschka, E.R., Hruschka Jr, E.R.: Tweet sentiment analysis with classifier ensembles. Decision support systems 66, 170–179 (2014)

[9] Steck, H.: Embarrassingly shallow autoencoders for sparse data. In: The World Wide Web Conference, pp. 3251–3257 (2019)

[10] Sagha, H., Cummins, N., Schuller, B.: Stacked denoising autoencoders for sentiment analysis: a review. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 7(5), 1212 (2017)

[11] Liang, H., Sun, X., Sun, Y., Gao, Y.: Text feature extraction based on deep learning: a review. EURASIP journal on wireless communications and networking 2017(1), 1–12 (2017)

[12] Dahlgaard, S., Knudsen, M., Thorup, M.: Practical hash functions for similarity estimation and dimensionality reduction. Advances in Neural Information Processing Systems 30 (2017)

[13] Valdenegro-Toro, M., Pincheira, H.: Implementing noise with hash functions for graphics processing units. arXiv preprint arXiv:1903.12270 (2019)

[14] Cheng, M., Wu, Y., Zhou, X., Li, J., Zhang, L.: Efficient web archive searching (2020)

[15] Cho, K., Van Merri¨enboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y.: Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078 (2014)

[16] An, J., Cho, S.: Variational autoencoder based anomaly detection using reconstruction probability. Special Lecture on IE 2(1), 1–18 (2015)

[17] Liu, G., Xie, L., Chen, C.-H.: Unsupervised text feature learning via deep variational auto-encoder. Information Technology and Control 49(3), 421–437 (2020)

[18] Zhang, Z., Zhai, S.: Semisupervised autoencoder for sentiment analysis. Google Patents. US Patent 11,205,103 (2021)

[19] Akhtar, M.S., Sawant, P., Sen, S., Ekbal, A., Bhattacharyya, P.: Solving data sparsity for aspect based sentiment analysis using cross-linguality and multi-linguality. (2018). Association for Computational Linguistics

[20] Ye, D., Li, Y., Tao, C., Xie, X., Wang, X.: Multiple feature hashing learning for large-scale remote sensing image retrieval. ISPRS International Journal of Geo-Information 6(11), 364 (2017)

[21] Guibon, G., Ermakova, L., Seffih, H., Firsov, A., Le No´e-Bienvenu, G.: Multilingual fake news detection with satire. In: CICLing: International Conference on Computational Linguistics and Intelligent Text Processing (2019)

[22] Seger, C.: An investigation of categorical variable encoding techniques in machine learning: binary versus one-hot and feature hashing (2018)

[23] Kirasich, K., Smith, T., Sadler, B.: Random forest vs logistic regression: binary classification for heterogeneous datasets. SMU Data Science Review 1(3), 9 (2018)
