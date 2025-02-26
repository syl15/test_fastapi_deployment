Ideally be able to swap out dummy model for classification model or sentiment analysis 

But still be able to use the API the same. 

Can just use a rule based classifier? 

def dummy_disaster_classifier(text: str) -> str:
    disaster_keywords = ["fire", "earthquake", "flood", "hurricane", "explosion"]
    if any(word in text.lower() for word in disaster_keywords):
        return "Disaster"
    return "Not a Disaster"

that function would kinda be like the model you need to host?

for this story, instead of loading a keras model/creating functions to preprocess or predict, we would just have our one dummy disaster classifier function. But the rest of the api setup should be the same i believe