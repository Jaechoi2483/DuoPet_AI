# services/video_recommend/keyword_extractor.py

from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords(text: str, top_n: int = 3):
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    return [kw for kw, _ in keywords]
