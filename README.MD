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
