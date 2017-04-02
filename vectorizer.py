from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
import os

cur_dir = os.path.dirname(__file__)
stop = joblib.load(open(os.path.join(cur_dir, "pkl_objects", "stopwords.pkl"), "rb"))


def tokenizer(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[^a-zA-Z]", " ", text.lower()) + " ".join(emoticons).replace("-", " ")
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


vect = HashingVectorizer(decode_error="ignore",
                         n_features=2 ** 21,
                         preprocessor=None,
                         tokenizer=tokenizer)


