# Misadventures with Neural Networks and Settling on Naive-Bayes

## Task

The task is to take a heterogenous data set, a mixture of documents which are movie reviews and some which are not, and implepent a system of ternary classificiation. Non-reviews recieve the tag 0, postive movie reviews recieve the tag 1, and negative 2. So really, there are two different classification tasks going on here. The first is to discriminate between reviews and non-reviews. The second is to tell apart positive and negative reviews. And of course, the model should have a high F1 and accuracy score, while still generalizing well.

## The Approach That Didn't Work

```python3
def distilbert_sent(text):
    
    threshold_min = 0.540 # is it possible to find a range of values between positive and negative we can use to classify neutral?
    threshold_max = 0.542

    sentiment_labels = {
    0: "neutral",
    1: "postive",
    2: "negative"
}

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs).logits
    predicted_label = torch.argmax(outputs, dim=1)
    predicted_score = float(torch.softmax(outputs, dim=1)[0][predicted_label].item())
    
    if predicted_score > threshold_min and predicted_score < threshold_max:
        return (sentiment_labels[0], predicted_score)
    elif predicted_score > threshold_max:
        return (sentiment_labels[1], predicted_score)
    else:
        return (sentiment_labels[2], predicted_score)
```
or this:
```python3
samples = df.TEXT.to_list()[:1000]
values = []
for sample in samples:
    sent_value = distilbert_sent(sample)[1]
    values.append(sent_value)

average = np.mean(values) # Could using an average of the values as a threshold work?

def distilbert_sent2(text):
    
    threshold = average  # is it possible to find a range of values between positive and negative we can use to classify neutral?
    

    sentiment_labels = {
    0: "neutral",
    1: "postive",
    2: "negative"
}

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs).logits
    predicted_label = torch.argmax(outputs, dim=1)
    predicted_score = float(torch.softmax(outputs, dim=1)[0][predicted_label].item())
    
    if isclose(average,predicted_score) == True:
        return (sentiment_labels[0], predicted_score)
    elif predicted_score > threshold:
        return (sentiment_labels[1], predicted_score)
    else:
        return (sentiment_labels[2], predicted_score)

```
I won't bother to show the whole notebook containing this spiral into insanity, as it didn't work and was ultimately a waste of my time. But basically, the idea was to use a binary classifier, run it on the data set, take the confidence values it returns, and either use the average of the confidence values as a threshold to identify a neutral text, or use use a minimum and maximum threshold to classify a "neutral" text. A fatal flaw in this approach is my false belief that there were neutral movie reviews in the data set, which was not the case. That's what I get for reading the directions once a month ago then never looking again. Despite this method not working, I felt compelled to mention it, since I spent most of my time one it, fruitlessly.

## The Approach That *Did* Work

Pressed for time, what I wound up doing was creating a good 'ol Naive Bayes model, as I was worried I wouldn't have time to finetune a regression based model. There is really nothing special about the model itself, just the vanilla NB model from Sci-Kit Learn's API. The part of the approach that does involve a good deal of iteration on my part is the data preprocessing, so I will discuss that in more depth.

I knew that to get better performance with Naive Bayes, I'd need to look carefully at how the data is tokenized. The first thing I did to address this was use NLTK's TweetTokenizer. Why? I intuit that the domains of Tweets and online movie reviews likely have much in common. This tokenizer accounts for irregularities that may appear in casual, short form prose posted online. Therefore, it was my suspicion that this tokenizer built specifically for this purpose would offer better performance that just splitting the data in a more generic fashion.

Next, I decided to de-case and lemmatize the text. Case may be a usefule feature for sentiment analysis of datsets with with lots of ANGRY ALL CAPS, but I don't think that outweighs normal use of case in this dataset. So, it seemed better to normalize the case. As for lemmatization, here was my approach:

```python3
pos_tags = pos_tag(tokens)
        for each_token, tag in pos_tags: # Normalize tags to feed to lemmatizer
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            elif tag.startswith('RB'):
                pos ='r'
            else:
                pos = 'a' # I don't think the distinction between satellite adjectives and adjectives is very meaningful here
            lemmatized_token = lemmatizer.lemmatize(each_token, pos)
            lemmatized_tokens.append(lemmatized_token)

```

I first ran the text through NLTK's POS tagger, then converted the POS tags to tags which are accepted by NLTK's WordNetLemmatizer. I wasn't quite sure how to divine from the Penn Tree Bank tags what was a satellite object and what wasn't (or if it even mattered), so I just labeled all adjectives as 'a'. Why lemmatize? The vectors created in this approach were initially very sparse. Lemmatizing shirnks the vocabulary, but in a systematic way that still leaves words important for sentiment intact, leading to denser vectors and better performance and accuracy.

For vectorization, I chose a TF-IDF approach. it not only focuses on the frequency of words present in the corpus but also provides the importance of the words in context, which is ideal for a classification task like this. TF-IDF allows us to get around some of the flaws of Bag of Words in this way by removing less important tokens from our analysis. To build on this effect, I also included a stopwords list. I initially tried NLTK's, but it was too expansive and actually hurt the accuracy of the model. So, I implemented a smaller, more generic list of things like pronouns, copula forms, etc.. Finally, I decided to go with trigram vectors, as i found these offered optimal accuracy and performance after testing out bigrams and tetragrams as well.

This is how I broke down my training and validation set:

```python3
# Break data down into a training set and a validation set
x_train, x_valid, y_train, y_valid= train_test_split(df['TEXT'].astype(str), df['LABEL'], test_size=0.1, random_state=7)

```

I experimented with setting 0.2 as the test size, but it hurt performance. After this, it was simply a matter of passing the data into the preprocessing pipeline, and then the NB model itself.

## Results

Since I didn't know which parameter (weighted, macro, or micro) was optimal to use when running the Scikit-Learn metrics for this task, I'm showing all 3, in addition to the "meta" versions of the F1 and precison scores, which just means I averaged all 3 versions of the scores together, because why not?

These are the final results for my solution:

**Weighted**

F1: 0.9130491924253523

Precision: 0.9140106845361778

Accuracy: 0.9127986348122867

**Macro**

F1: 0.9018131201839136

Precision: 0.9005630795842544

Accuracy: 0.9127986348122867

**Micro**

F1: 0.9127986348122867

Precision: 0.9127986348122867

Accuracy: 0.9127986348122867

**Meta**

F1 Meta: 0.9092203158071842

Precision Meta: 0.9091241329775729

Accuracy Meta: 0.9127986348122867

The model returns roughly 91% percent in all 3 metrics, which isn't bad considering how simple my approach is. Also, it only takes about 4 minutes to train this model on an M1 MacBook Air, so speed is not an issue here.

```python3
0      1      2
0[7732,  212,  184]
1[ 190, 4075,  538]
2[  53,  356, 4240]

```

Another interesting thing to look at is the confusion matrix above, which isn't aligned quite properly because I can't figure out how to tell markdown to do it, but it's still readable. In the test file, there is a roughly equal split of label counts between the the positive (1) and negative (2) categories, making up about half the data set. The other half is labeled as neutral (0). It is interesting to note that the model performs worst when predicting postive (1) texts with 728 false predictions. The performance for the neutral and negative categories was roughly equal with 396 and 409 false predictions, respectively. For future optimizations, it may be productive to analyze why so many postive documents are coming up as negative.