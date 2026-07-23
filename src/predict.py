def predict(text: str) -> str:
    """
    Placeholder version — replace with real model later.
    Flags messages containing common spam trigger words.
    """
    spam_keywords = ["free", "win", "winner", "urgent", "call now", "click"]
    text_lower = text.lower()
    if any(word in text_lower for word in spam_keywords):
        return "spam"
    return "ham"
