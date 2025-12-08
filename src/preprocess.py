import string
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ---- SSL workaround (mainly for some environments) ----
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    # Not needed everywhere
    pass

_NLTK_READY = False


def ensure_nltk_resources():
    """
    Ensure that required NLTK resources are available.
    Safe to call multiple times – it will only download once per process.
    """
    global _NLTK_READY
    if _NLTK_READY:
        return

    # punkt sentence/word tokenizer
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # newer NLTK also uses punkt_tab
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        # Often covered by 'punkt', but just in case:
        nltk.download("punkt", quiet=True)

    # stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    # touch the resource to be sure
    _ = stopwords.words("english")

    _NLTK_READY = True


def transform_text(text: str) -> str:
    """
    Clean and normalize raw SMS text.

    Steps:
    - ensure NLTK data (punkt, stopwords)
    - lowercase
    - tokenize
    - keep only alphanumeric tokens
    - remove stopwords and punctuation
    - apply Porter stemming
    """
    ensure_nltk_resources()

    # lowercase
    text = text.lower()

    # tokenize
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # Fallback: very simple split if something goes wrong
        tokens = text.split()

    # keep only alphanumeric
    y = [tok for tok in tokens if tok.isalnum()]

    # remove stopwords and punctuation
    filtered = [
        tok
        for tok in y
        if tok not in stopwords.words("english") and tok not in string.punctuation
    ]

    # stemming
    stemmed = [ps.stem(tok) for tok in filtered]

    return " ".join(stemmed)
