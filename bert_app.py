import streamlit as st
from transformers import BertTokenizer, BertModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import json
import base64

# Load BERT and sentiment analysis models
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    sentiment_model = pipeline("sentiment-analysis")
    return tokenizer, model, sentiment_model

tokenizer, model, sentiment_model = load_bert_model()

# Predefined Q&A
qa_pairs = {
    "What is your name?": "I am a chatbot powered by BERT!",
    "How are you?": "I'm just a bunch of code, but I'm doing great!",
    "What is BERT?": "BERT stands for Bidirectional Encoder Representations from Transformers. It’s a powerful NLP model.",
    "Tell me a joke.": "Why don't programmers like nature? It has too many bugs.",
}

@st.cache_data
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

@st.cache_data
def precompute_embeddings(qa_pairs):
    return {question: get_bert_embedding(question) for question in qa_pairs}

predefined_embeddings = precompute_embeddings(qa_pairs)

# Chatbot response function
def chatbot_response(user_input):
    user_embedding = get_bert_embedding(user_input)
    similarities = {
        question: cosine_similarity(user_embedding, predefined_embeddings[question])[0][0]
        for question in qa_pairs
    }
    best_match = max(similarities, key=similarities.get)

    if similarities[best_match] > 0.6:
        return qa_pairs[best_match]
    else:
        suggestions = [q for q, sim in similarities.items() if sim > 0.3]
        if suggestions:
            return f"I'm not sure about that. Did you mean: {', '.join(suggestions)}?"
        else:
            return "I'm not sure how to respond to that."

# Add new Q&A dynamically
def add_qa_pair(question, response):
    qa_pairs[question] = response
    predefined_embeddings[question] = get_bert_embedding(question)

# Translate text
translator = Translator()

def translate_text(text, target_language="en"):
    return translator.translate(text, dest=target_language).text

# CSS for styling
def set_custom_style():
    st.markdown(
        """
        <style>
        .chat-bubble-user {
            background-color: #e6f7ff;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            max-width: 60%;
            text-align: left;
        }
        .chat-bubble-bot {
            background-color: #ffe6e6;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            max-width: 60%;
            text-align: left;
        }
        .chat-container {
            max-width: 80%;
            margin: auto;
        }
        .stSidebar {
            background-color: #f8f9fa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_style()

# Streamlit UI
st.sidebar.title("Settings")
st.sidebar.subheader("Add a New Q&A")
new_question = st.sidebar.text_input("New Question:")
new_response = st.sidebar.text_input("New Response:")
if st.sidebar.button("Add Q&A"):
    if new_question and new_response:
        add_qa_pair(new_question, new_response)
        st.sidebar.success("New Q&A added!")
    else:
        st.sidebar.error("Both question and response are required.")

st.sidebar.subheader("Multilingual Support")
enable_translation = st.sidebar.checkbox("Enable Translation")
target_language = st.sidebar.selectbox("Target Language", ["en", "es", "fr", "de", "zh"], index=0)

st.title("💬 Advanced BERT Chatbot")
st.write("A chatbot powered by BERT with enhanced UI and features. Ask me anything!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat section
user_input = st.text_input("You:", placeholder="Type your message here...")
if user_input:
    if enable_translation:
        user_input = translate_text(user_input, target_language="en")

    response = chatbot_response(user_input)
    sentiment = sentiment_model(user_input)[0]["label"]

    # Log chat
    st.session_state.chat_history.append({"user": user_input, "bot": response, "sentiment": sentiment})

# Display chat history
for message in st.session_state.chat_history:
    st.markdown(f'<div class="chat-bubble-user">🧑‍💻: {message["user"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble-bot">🤖: {message["bot"]} <br> <small>Sentiment: {message["sentiment"]}</small></div>', unsafe_allow_html=True)
