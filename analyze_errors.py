def analyze(question):
    """Analyze the type of question:
        - who
        - what
        - where
    """
    words = question.lower().split()
    for keyword in ["what","who","when","which"]:
        if keyword in words:
            return keyword
    
    return "--"
