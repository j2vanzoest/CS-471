#Naive Bayes: Classify thing as Spam or Ham 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#load the dataset
data = pd.read_csv('SpamDetection.csv', encoding='latin-1')
data = data.rename(columns={'Target': 'label', 'data': 'message'})

#P(spam | sentence) = P(sentence | spam) x p(spam) 
# Use the first 20 samples for training
train_data = data.iloc[:20].copy()
#P(ham |sentence) = P(sentence | ham) x p(ham)
# Use the remaining 10 samples for testing
test_data = data.iloc[20:30].copy()

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))

#Compute the probabilites 
spam_count = (train_data['label'] == 'spam').sum()
ham_count = (train_data['label'] == 'ham').sum()
p_spam = spam_count / len(train_data)
p_ham = ham_count / len(train_data)


#Convert the messages into lowercase word lists so they can be processed
def tokenize(sentence):
    return re.findall(r"[a-z]+", sentence.lower())

#Count words in the dataset 
spam_words = {}
ham_words = {}

#for the spam 
for sentence in train_data[train_data['label'] == 'spam']['message']:
    for word in tokenize(sentence):
         spam_words[word] = spam_words.get(word, 0) + 1

#for the ham
for sentence in train_data[train_data['label'] == 'ham']['message']:
    for word in tokenize(sentence):
         ham_words[word] = ham_words.get(word, 0) + 1

# Create vocabulary using training data
vocab = set(spam_words) | set(ham_words)
vocab_size = len(vocab)

total_spam_words = sum(spam_words.values())
total_ham_words = sum(ham_words.values())


# Conditional probabilities: P(word | spam) and P(word | ham)
# Laplace smoothing prevents zero probabilities

p_word_given_spam = {}
p_word_given_ham = {}

for word in vocab:
    p_word_given_spam[word] = (
        spam_words.get(word, 0) + 1
    ) / (
        total_spam_words + vocab_size
    )

    p_word_given_ham[word] = (
        ham_words.get(word, 0) + 1
    ) / (
        total_ham_words + vocab_size
    )


print("P(spam):", p_spam)
print("P(ham):", p_ham)


def sentence_probability(sentence, word_probabilities):
    """Calculate P(sentence | class)."""
    probability = 1.0

    for word in tokenize(sentence):
        # Use smoothing for words not seen during training
        probability *= word_probabilities.get(
            word,
            1 / (vocab_size + 1)
        )

    return probability


def sentence_log_probability(sentence, word_probabilities):
    """
    Calculate log(P(sentence | class)).
    Log probabilities prevent numerical underflow.
    """
    log_probability = 0.0

    for word in tokenize(sentence):
        word_probability = word_probabilities.get(
            word,
            1 / (vocab_size + 1)
        )

        log_probability += np.log(word_probability)

    return log_probability


def classify_sentence(sentence):
    # P(sentence | spam) and P(sentence | ham)
    p_sentence_given_spam = sentence_probability(
        sentence,
        p_word_given_spam
    )

    p_sentence_given_ham = sentence_probability(
        sentence,
        p_word_given_ham
    )

    # Use log probabilities for reliable classification
    spam_log_score = (
        np.log(p_spam)
        + sentence_log_probability(sentence, p_word_given_spam)
    )

    ham_log_score = (
        np.log(p_ham)
        + sentence_log_probability(sentence, p_word_given_ham)
    )

    # Convert log scores into normalized probabilities
    largest_score = max(spam_log_score, ham_log_score)

    spam_score = np.exp(spam_log_score - largest_score)
    ham_score = np.exp(ham_log_score - largest_score)

    total_score = spam_score + ham_score

    p_spam_given_sentence = spam_score / total_score
    p_ham_given_sentence = ham_score / total_score

    if p_spam_given_sentence > p_ham_given_sentence:
        predicted_class = "spam"
    else:
        predicted_class = "ham"

    return (
        p_sentence_given_spam,
        p_sentence_given_ham,
        p_spam_given_sentence,
        p_ham_given_sentence,
        predicted_class
    )


# Classify every sentence in the test set
correct_predictions = 0

print("\nTest Set Results")
print("=" * 70)

for _, row in test_data.iterrows():
    sentence = row["message"]
    actual_class = row["label"]

    (
        p_sentence_given_spam,
        p_sentence_given_ham,
        p_spam_given_sentence,
        p_ham_given_sentence,
        predicted_class
    ) = classify_sentence(sentence)

    print("Sentence:", sentence)
    print("P(sentence | spam):", p_sentence_given_spam)
    print("P(sentence | ham):", p_sentence_given_ham)
    print("P(spam | sentence):", p_spam_given_sentence)
    print("P(ham | sentence):", p_ham_given_sentence)
    print("Predicted class:", predicted_class)
    print("Actual class:", actual_class)
    print("-" * 70)

    if predicted_class == actual_class:
        correct_predictions += 1


# Calculate test-set accuracy
total_test_sentences = len(test_data)
accuracy = correct_predictions / total_test_sentences

print("\nAccuracy Results")
print("Correct predictions:", correct_predictions)
print("Total test sentences:", total_test_sentences)
print("Test accuracy:", accuracy)
print("Test accuracy percentage:", accuracy * 100, "%")