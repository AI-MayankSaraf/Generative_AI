import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import base64
from transformers import pipeline
import json
from googletrans import Translator

# Add background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Uncomment the next line to set the background
# set_background(r"img.jpg")

# Load BERT tokenizer and model with caching
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    sentiment_model = pipeline("sentiment-analysis")
    return tokenizer, model, sentiment_model

tokenizer, model, sentiment_model = load_bert_model()

# Predefined questions and responses
qa_pairs = {
    "What is your name?": "I am a chatbot powered by BERT!",
    "How are you?": "I'm just a bunch of code, but I'm doing great!",
    "What is BERT?": "BERT stands for Bidirectional Encoder Representations from Transformers. It’s a powerful NLP model.",
    "Tell me a joke.": "Why don't programmers like nature? It has too many bugs.",
}

# Function to get BERT embeddings
@st.cache_data
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Precompute embeddings for predefined questions
@st.cache_data
def precompute_embeddings(qa_pairs):
    return {question: get_bert_embedding(question) for question in qa_pairs}

predefined_embeddings = precompute_embeddings(qa_pairs)

# Sentiment analysis function
def detect_sentiment(text):
    sentiment = sentiment_model(text)
    return sentiment[0]["label"]

# Function to get the chatbot's response
def chatbot_response(user_input):
    user_embedding = get_bert_embedding(user_input)
    similarities = {
        question: cosine_similarity(user_embedding, predefined_embeddings[question])[0][0]
        for question in qa_pairs
    }
    best_match = max(similarities, key=similarities.get)

    # Provide suggestions if no strong match is found
    if similarities[best_match] > 0.6:
        return qa_pairs[best_match]
    else:
        suggestions = [q for q, sim in similarities.items() if sim > 0.3]
        if suggestions:
            return f"I'm not sure about that. Did you mean: {', '.join(suggestions)}?"
        else:
            return "I'm not sure how to respond to that."

# Dynamic Q&A updates
def add_qa_pair(question, response):
    qa_pairs[question] = response
    predefined_embeddings[question] = get_bert_embedding(question)

# Multilingual Support
translator = Translator()

def translate_text(text, target_language="en"):
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Logging conversations
def log_conversation(user_input, chatbot_response):
    with open("chat_log.json", "a") as log_file:
        log_file.write(json.dumps({"User": user_input, "Chatbot": chatbot_response}) + "\n")

# Streamlit UI
st.title("Advanced BERT Chatbot")
st.write("A BERT-powered chatbot with enhanced features like dynamic Q&A updates, sentiment analysis, and multilingual support.")
st.subheader("Ask me anything!")

# User input and response
user_input = st.text_input("You:", placeholder="Type your message here...")
if user_input:
    # Translate if necessary
    if st.checkbox("Translate to English"):
        user_input = translate_text(user_input, target_language="en")

    # Sentiment analysis
    sentiment = detect_sentiment(user_input)

    # Generate response
    response = chatbot_response(user_input)

    # Log the conversation
    log_conversation(user_input, response)

    # Display response and sentiment
    st.write(f"**Chatbot:** {response}")
    st.write(f"*Sentiment detected: {sentiment}*")

# Add new Q&A
st.subheader("Add a New Question and Response")
new_question = st.text_input("New Question:")
new_response = st.text_input("New Response:")
if st.button("Add Q&A"):
    if new_question and new_response:
        add_qa_pair(new_question, new_response)
        st.success("New Q&A pair added successfully!")
    else:
        st.error("Both question and response must be provided!")

# Footer
st.markdown("---")
