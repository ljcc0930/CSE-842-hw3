# CSE-842-hw3
## Problem 1
### a. Package used

PyTorch (through the Hugging Face Transformers library)

### b. Model used

distilbert-base-uncased

### c. Brief description of the model

DistilBERT is a smaller and faster version of the original BERT model. It retains 95% of the performance while being 60% smaller and 2.5 times faster. The 'distilbert-base-uncased' model is based on the BERT 'base' architecture with 12 transformer layers, 768 hidden units, and 12 attention heads. The model is trained on lowercase text, hence the 'uncased' part in the name.

### d. Results

The training logs show that the model's performance improved over the course of 3 epochs. The training loss decreased from 1.6176 to 1.1103. The evaluation loss also dropped, going from 1.5995 to 1.1106. The final evaluation loss is 1.1106, which indicates that the model's performance on the test data improved after fine-tuning on the training data. Please note that the performance metrics such as accuracy or F1 score are not provided in the logs, but you can obtain them using the `trainer.evaluate()` method.

Here's a summary of the loss values in a Markdown table:

| Epoch | Training Loss | Evaluation Loss |
|-------|---------------|-----------------|
| 1     | 1.6211        | 1.5995          |
| 2     | 1.4139        | 1.3510          |
| 3     | 1.1103        | 1.1106          |

### e. Something new or interesting learned during implementation

During this implementation, I learned how to fine-tune a pre-trained DistilBERT model on a custom dataset (Yelp reviews) using the Hugging Face Transformers library. The code demonstrates how to load a dataset, preprocess it, set up training arguments, and train and evaluate the model using the Trainer class. It also shows how to use a custom tokenizer and apply it to the dataset using the map function.


## Problem 2

### Instructions
1. Install the necessary libraries:
```
pip install numpy pandas scikit-learn nltk gensim allennlp
```
3. Download the Word2Vec model from [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) and the GloVe model from glove.6B.300d.txt.
4. Update the data_dir variable in the code to the path of the Polarity 2.0 dataset, and update the paths for the Word2Vec and GloVe model files.
5. Run the script:
```
python SVC_polarity.py
```

### a. Your chosen NLP task and why you wanted to study it.
I chose sentiment analysis as my NLP task because it is a widely applicable problem in various domains, such as customer reviews, social media, and marketing analysis. Understanding the sentiment of a text can help businesses and organizations make better decisions and gain valuable insights into their customers and target audience.



### b. The dataset you are using and why you chose it for this task.
I chose the polarity 2.0 dataset for this task because it's a relatively smaller dataset, which makes it more suitable for models that are not deep neural networks to learn and reduce the performance gap. Using a smaller dataset can also highlight the potential risk of overfitting in large language models (LLMs) as they might be more prone to memorizing the training data rather than generalizing well to unseen data.

### c. The model(s) you used.
I used a Support Vector Machine (SVM) classifier with different word embeddings, including Word2Vec and GloVe. SVM is a popular choice for text classification tasks because it can efficiently handle high-dimensional input data and is less prone to overfitting.

### d. The features you tested.
I tested various word embeddings as features, such as Word2Vec and GloVe. These embeddings help capture semantic information in the text and are widely used in NLP tasks. The goal was to understand the impact of different embeddings on the performance of the SVM classifier.

### e. Link to the most recent, best-performing BERT (or related) model for this task.
[BERT for Sentiment Analysis](https://arxiv.org/pdf/1810.04805.pdf). The table summary of their results is as follows:

|Model| 	Accuracy|
|---|---|
|BERT(Base)	|0.935|
|BERT(Large)|	0.949|
|BERT(fine-tuned by me)| 0.8675 |

### f. Summary table of your results (precision, recall, F1, accuracy).
| Embedding   | Best Parameters       | Accuracy | Precision | Recall | F1-score |
|-------------|-----------------------|---------------------|-----------|--------|----------|
| Word2Vec    | C=10, linear kernel | **0.8550**              | 0.86    | 0.85 | 0.86   |
| GloVe-300   | C=10, linear kernel | 0.8225              |	0.82 |	0.82 |	0.82   |
| GloVe-50    | C=1, linear kernel | 0.7475              | 0.75    | 0.74 | 0.75   |


### g. Brief discussion of:
**i. where you think your modeling approach and features worked well**

The use of word embeddings as features captures semantic information about the words in the movie reviews, which is essential for sentiment analysis. SVM is a powerful classifier and can handle high-dimensional data, which is suitable for this task, and I use `GridSearchCV` for my parameters to get the best performance over cross validations to further improve my method.

**ii. where you might have been able to improve**

The code could benefit from feature engineering, such as including n-grams, part-of-speech tags, or other linguistic features. Additionally, other models, such as logistic regression or ensemble methods, could be tested to compare performance.

**iii. why you think your model did or did not outperform the BERT model of (e).**

It is likely that the provided model did not outperform the BERT model because BERT is a more advanced model, specifically designed for NLP tasks, and it leverages a deep learning architecture with attention mechanisms. BERT has been pre-trained on a large corpus and fine-tuned for specific tasks, which allows it to capture more nuanced patterns and dependencies in the text, resulting in better performance for many NLP tasks, including sentiment analysis.

The provided model, on the other hand, relies on simpler word embeddings and a Support Vector Machine classifier. Although the embeddings capture some semantic information, they may not be as powerful as BERT's contextualized representations. Moreover, the SVM classifier, although effective in many situations, may not be able to capture complex patterns and relationships as effectively as the deep learning architecture used in BERT.

To potentially close the performance gap between the provided model and the BERT model, one could experiment with more advanced word embeddings, such as contextualized embeddings from models like ELMo or BERT itself. Additionally, experimenting with other classification models or ensemble methods could lead to performance improvements. Lastly, incorporating more advanced feature engineering techniques, such as n-grams, part-of-speech tags, or other linguistic features, may enhance the model's ability to capture the sentiment in the text.

### h. Optional for 5 bonus points: Add BERT contextual embeddings into your features
| Embedding   | Best Parameters       | Accuracy | Precision | Recall | F1-score |
|-------------|-----------------------|---------------------|-----------|--------|----------|
| Bert    | C=5, rbf kernel | 0.8225              | 0.82    | 0.82 | 0.82   |
```
python SVC_polarity_bertembed.py
```

### h. Optional for 5 bonus points: Add BERT contextual embeddings into your features
Trained with BERT embedding. Accuracy: 0.4875
```
python bilstm.py
```

## Problem 3

The finetuned performance is summarized in Problem 2e.
```
python finetune_bert_polarity.py
```