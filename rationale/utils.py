import time 
import datetime
import pytz
import re

def get_timestamp():
    "Store a timestamp for when training started."
    timestamp = time.time()
    timezone = pytz.timezone("Etc/GMT+7")
    dt = datetime.datetime.fromtimestamp(timestamp, timezone)
    return dt.strftime("%Y-%m-%d:%H:%m:%S")

def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text

def strict_accuracy(gt: dict, pred: dict) -> dict:
    gt_verdict = gt["verdict"]
    pred_verdict = pred["verdict"]
    gt_evidence = gt["evidence"]
    pred_evidence = pred["evidence"]

    gt_evidence = preprocess_text(gt_evidence)
    pred_evidence = preprocess_text(pred_evidence)

    acc = int(gt_verdict == pred_verdict)
    acc_1 = int(gt_evidence == pred_evidence)
    strict_acc = acc * acc_1

    return {
        "strict_acc": strict_acc,
        "acc": acc,
        "acc@1": acc_1,
    }

def flatten(z):
    return [x for y in z for x in y]

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def transform_sent(sentence):
	sentence = convert_to_unicode(sentence)
	sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
	sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
	sentence = re.sub("-LRB-", "(", sentence)
	sentence = re.sub("-RRB-", ")", sentence)
	sentence = re.sub("-COLON-", ":", sentence)
	sentence = re.sub("_", " ", sentence)
	sentence = re.sub("\( *\,? *\)", "", sentence)
	sentence = re.sub("\( *[;,]", "(", sentence)
	sentence = re.sub("--", "-", sentence)
	sentence = re.sub("``", '"', sentence)
	sentence = re.sub("''", '"', sentence)
	return sentence

    