import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pickle

dataset = pd.read_csv('spam.csv', encoding='latin1')
dataset.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, axis=1)
dataset.columns = ["Category", "Message"]
dataset.drop_duplicates(keep='first', inplace=True)

number_of_spam = dataset[dataset['Category'] == 'spam'].shape[0]
number_of_ham = dataset[dataset['Category'] == 'ham'].shape[0]

plt.figure(figsize=(15, 6))
mail_categories = [number_of_ham, number_of_spam]
labels = [f"Ham = {number_of_ham}", f"Spam = {number_of_spam}"]
explode = [.2, 0]
plt.pie(mail_categories, labels=labels, explode=explode, autopct="%.2f %%")
plt.title("Ham vs Spam")
plt.show()

encoder = LabelEncoder()
dataset['spam'] = encoder.fit_transform(dataset['Category'])

x = dataset['Message']
y = dataset['spam']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
x_train_counts = vectorizer.fit_transform(x_train)
classifier = MultinomialNB()
classifier.fit(x_train_counts, y_train)

# Save trained classifier to a pickle file
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Save fitted CountVectorizer to a pickle file
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

x_test_counts = vectorizer.transform(x_test)

# Display the confusion matrix
y_pred = classifier.predict(x_test_counts)
confusion_matrix = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Ham", "Spam"])
cm_display.plot()
plt.show()

print(classification_report(y_test, y_pred))

emails = [
    "Hey jessica, I'm at the Ms.Salahshor class waiting for you, where are you?",
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!',
    '''Join us on Saturday, February 24 at 14:00 UTC on our YouTube channel to take this
    interactive lesson, taught by Tutor Darryl.'''
]

emails_count = vectorizer.transform(emails)
print(emails_count)
print(classifier.predict(emails_count))
