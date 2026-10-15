import random        #for choosing random responses in the end
import json
import pickle        #for serialization
import numpy as np 

import nltk   #natural language toolkit
from nltk.stem import WordNetLemmatizer     #this wordNetLemmatizer is going to reduce the word to its stem .

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intense.json", encoding="utf-8").read())  #utf-8 is a character encoding format. It tells Python how to read text stored in your intense.json file.
#this is reading the content of jsonfile as text..we are passing that text to the loads function and as result
#we get the json object which is a dictionary essentially in python

#we will create 3 empty list

words = []
classes = []
documents = []

ignore_letters = ["?", "!", ".", ",", ""]      # this is going to ignore the letter 






#now we have to iterte over the intent 

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])


# print(documents)



#let's do lemmatizer

words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_letters
]

words = sorted(set(words))
classes = sorted(set(classes))    #set eliminate duplicates , sorted turns it back into a list and sorts it
                     
# print(words)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))




#now we are getting in to machine learning part
#for this we need to understand why we're actually doing ...so problem here is right now that we have lots of character and words and classes and all of that... but those are not numerical values, we don't have anything we can feed intoneural network a neural network needs numerical values we cannot just feed words into it and expect it to work on its own...we need to represent these words as numerical values
#and in order to do that we are going to use something called "bag of words", we are going to set the individual words indicies or the individual word values to eighter zero or one depending on if it's occuring in that particular pattern and then we are going to also do the same thing for the classes 



training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [
        lemmatizer.lemmatize(word.lower())
        for word in word_patterns
    ]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])






#now  let's shuffle the data


random.shuffle(training)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])



#now we can start to build neural network model


model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = "softmax"))

sgd = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)


# Build the neural network

model = Sequential()

model.add(Dense(128, input_shape=(train_x.shape[1],), activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(train_y.shape[1], activation="softmax"))


# Compile the model

sgd = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)

# Train the model
model.fit(
    train_x,
    train_y,
    epochs=200,
    batch_size=5,
    verbose=1
)

# Save the trained model
hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)
model.save("chatbot_model.keras")

print("Training completed successfully!")



#now we will create a chatbot application itself that uses the trained model
