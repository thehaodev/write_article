from nltk.util import ngrams
from collections import Counter
from preprocess import stem_lema

def count_word(text, n=2):
    """
    Hàm tính n-gram profile của văn bản sau tiền xử lý.
    
    Args:
        text (str): Văn bản đầu vào
        n (int): Giá trị n của n-gram (mặc định là bigram)
        
    Returns:
        Counter: Bộ đếm tần suất các n-gram
    """
    tokens = stem_lema(text)  # Tiền xử lý văn bản
    n_gram_list = list(ngrams(tokens, n))
    profile = Counter(n_gram_list)
    
    return profile