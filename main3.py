import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import pyjokes
import pywhatkit
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pyowm
import geocoder
from newsapi import NewsApiClient
import vertexai
from vertexai.language_models import TextGenerationModel
import re
import pandas as pd
import random
import requests
# Initialize API
owm = pyowm.OWM('ea1c27935c886d9b6b465e2f23a44b0c')
newsapi = NewsApiClient(api_key='b8a263fc824049a692f4aed478f1bd96')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\\Ninad\\SeventhProject\\verdant-descent-419814-f0efe3d56838.json"

# Initialize BERT tokenizer and model

model_path = "D:/Ninad/SeventhProject/intent_model-copy-1.pth"
model_state_dict = torch.load(model_path)


class IntentModel(nn.Module):
    def __init__(self, bert_model):
        super(IntentModel, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        logits = outputs.logits
        return logits


model = IntentModel(
    BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7))
model.load_state_dict(model_state_dict)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("intent_model.pth", num_labels=6)  # Update num_labels based on your dataset
# model_state_dict = torch.load('intent_model.pth')
# model.load_state_dict(model_state_dict)

# Define RNN model


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

    # defining methods


def get_weather():
    try:
        location = geocoder.ip('me')
        latitude, longitude = location.latlng
        observation = owm.weather_manager().weather_at_coords(latitude, longitude)
        weather = observation.weather
        temperature = weather.temperature('celsius')['temp']
        status = weather.detailed_status
        return f"The weather in {location.city} is {status} with a temperature of {temperature} degrees Celsius."
    except pyowm.exceptions.api_response_error.NotFoundError:
        return "Sorry, the city name was not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_latest_news():
    try:
        top_headlines = newsapi.get_top_headlines(
            language='en', country='in')
        articles = top_headlines['articles']
        news_headlines = []
        for idx, article in enumerate(articles[:5], start=1):
            source = article['source']['name']
            title = article['title']
            # description = article['description']
            news_headlines.append((source, title))
        return news_headlines
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def get_search_results(query):
    try:
        vertexai.init(project="verdant-descent-419814",
                      location="us-central1")
        parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.9,
            "top_p": 1
        }
        text_model = TextGenerationModel.from_pretrained("text-bison")
        response = text_model.predict(
            query,
            **parameters
        )
        lines = response.text.split('\n')[:2]

    # Join the selected lines back together
        selected_lines = '\n'.join(lines)
        return f"Response from Model: {selected_lines}"
    except Exception as e:
        return f"An unexpected error occured: {str(e)}"

    # Define voice assistant


class VoiceAssistant:
    def __init__(self):
        self.engine = pyttsx3.init('sapi5')
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[0].id)
        # Assuming input_size is the BERT output size
        self.rnn_model = RNNModel(
            # Update output_size based on your dataset
            input_size=768, hidden_size=128, output_size=7)

    def speak(self, audio):
        self.engine.say(audio)
        self.engine.runAndWait()

    def take_command(self, max_attempts=3):
        r = sr.Recognizer()
        attempts = 0
        while attempts < max_attempts:
            with sr.Microphone() as source:
                print("Listening...")
                r.pause_threshold = 1
                r.adjust_for_ambient_noise(
                    source, duration=1)  # Adjust for ambient noise
                audio = r.listen(source)
            try:
                print("Recognizing...")
                query = r.recognize_google(audio, language='en-in')
                print(f"User Said: {query} \n")
                # attempts = 3
                return query
            except Exception as e:
                print(e)
                self.speak("Can you please repeat")
                attempts += 1
        self.speak("Sorry couldn't understand after multiple attempts.")

    def preprocess_query(self, query):
        # Remove special characters and unnecessary whitespace
        processed_query = re.sub(r"[^a-zA-Z0-9\s]", "", query)
        processed_query = processed_query.strip()

        # Check if the query contains any OOV words
        for word in processed_query.split():
            if word not in tokenizer.vocab:
                # Replace the OOV word with a placeholder token
                processed_query = processed_query.replace(word, "[OOV]")

        return processed_query

    def classify_intent(self, query, confidence_threshold=0.5):
        # Preprocess the query to remove noise and irrelevant information
        processed_query = self.preprocess_query(query)

        # Tokenize the processed query
        inputs = tokenizer(processed_query, return_tensors="pt",
                           max_length=128, truncation=True)

        # Perform inference
        outputs = model(input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask)
        probs = torch.softmax(outputs, dim=1)
        max_prob, predicted_label = torch.max(probs, dim=1)

        # Check if the maximum confidence of the predicted intent is above the threshold
        if max_prob.item() >= confidence_threshold:
            intents = ['time', 'weather', 'search',
                       'joke', 'news', 'wikipedia', 'play']
            return intents[predicted_label.item()]
        else:
            return "uncertain"

    def generate_response(self, intent, query):
        if intent == "time":
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            strDate = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"The date is {strDate} and In 24 hours format the time is {strTime}"

        elif intent == "weather":
            # self.speak("Sure, please specify the city name.")
            # location = self.take_command().lower()
            return get_weather()

        elif intent == "news":
            self.speak("Top 5 headlines are:")
            news = get_latest_news()
            if isinstance(news, list):
                response = "Here are the latest news headlines:\n"
                for idx, (source, title) in enumerate(news, start=1):
                    response += f"According to {source}:"
                    response += f"Headline {idx}: {title}\n"
                    # response += f"Description: {description}\n"
                return response
            else:
                return news  # Handle error messages

        elif intent == "search":
            pywhatkit.search(query)
            search_media = get_search_results(query)
            return search_media

        elif intent == "wikipedia":
            info = wikipedia.summary(query, 2)
            # search_media = get_search_results(query)
            return info

        elif 'play' in query:
            song = query.replace('play', "")
            pywhatkit.playonyt(song)
            return f"playing {song}"

        elif intent == 'joke':
            response = requests.get("https://v2.jokeapi.dev/joke/Any")
            joke_data = response.json()
            if joke_data["type"] == "single":
                return joke_data["joke"]
            elif joke_data["type"] == "twopart":
                return f"{joke_data['setup']} ... {joke_data['delivery']}"

        elif intent == "uncertain":
            return "I am having trouble interpreting your request."
        else:
            return "How may I help you?"

    def start(self):
        self.speak("Initializing voice assistant...")
        self.speak("Hello! I am your voice assistant")
        hour = int(datetime.datetime.now().hour)
        if 0 <= hour < 12:
            self.speak("Good Morning")
        elif 12 <= hour < 18:
            self.speak("Good Afternoon")
        else:
            self.speak("Good Evening")
        self.speak("How may I help you?")
        # while True:
        query = self.take_command().lower()
        processed_query = self.preprocess_query(query)
        intent = self.classify_intent(processed_query)
        response = self.generate_response(intent, query)
        print(response)
        self.speak(response)


if __name__ == "__main__":

    assistant = VoiceAssistant()
    assistant.start()
