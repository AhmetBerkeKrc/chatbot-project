import warnings
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from datasets import load_dataset
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI as LangChainOpenAI
from twilio.rest import Client

# Disable warnings
warnings.filterwarnings("ignore")

# OpenAI API key 
api_key = "API_KEY"

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Load Sentence Transformer model for embeddings
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load FAISS index for similarity search
faiss_index = faiss.read_index("medical_faiss.index")

# Load medical dataset
dataset = load_dataset("codexist/medical_data")
texts = [row["data"] for row in dataset["train"]]

# Google Sheets API credentials 
credentials_file = "your_google_credentials_file.json"
sheet_id = "SHEET_ID"

# Google Sheets authorization and data retrieval
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
gspread_client = gspread.authorize(creds)
sheet = gspread_client.open_by_key(sheet_id)
worksheet = sheet.worksheet("app_sheet")

# Twilio API credentials 
twilio_account_sid = "SID"
twilio_auth_token = "TOKEN"
twilio_phone_number = "PHONE_NUMBER"

twilio_client = Client(twilio_account_sid, twilio_auth_token)

# Function to load data from Google Sheets
def load_data():
    try:
        values = worksheet.get_all_values()
        columns = values[0]
        df = pd.DataFrame(values[1:], columns=columns)
        return df
    except Exception as e:
        print(f"Error loading data from Google Sheets: {e}")
        return None

# Function to update appointment details in Google Sheets
def update_appointment(doctor_name, appt_time, user_name, user_id, user_phone, df):
    try:
        mask = (df["Doctor Name"] == doctor_name) & (df["Time"] == appt_time)
        df.loc[mask, "Patient Full Name"] = user_name
        df.loc[mask, "Patient ID Number"] = user_id
        df.loc[mask, "Patient Phone Number"] = user_phone
        df.loc[mask, "Status"] = "Booked"
        
        # Update the worksheet with the new data
        updated_values = [df.columns.tolist()] + df.values.tolist()
        worksheet.clear()
        worksheet.update(values=updated_values, range_name="A1")
        print(f"Appointment booked: Dr.{doctor_name} at {appt_time} for {user_name}")
    except Exception as e:
        print(f"Error updating appointment: {e}")

# Function to get available appointment slots
def get_available_appointments(df: pd.DataFrame):
    try:
        available_hours = df[df["Status"] == "Available"]
        return available_hours[["Doctor Name", "Date", "Time"]].to_string()
    except Exception as e:
        print(f"Error getting available appointments: {e}")
        return "Error retrieving available appointments."

# Function to book an appointment
def book_appointment(doctor_name: str, appt_time: str, user_name: str, user_id: str, user_phone: str, df: pd.DataFrame):
    try:
        mask = (df["Doctor Name"] == doctor_name) & (df["Time"] == appt_time)
        if not df.loc[mask].empty and df.loc[mask, "Status"].iloc[0] == "Available":
            update_appointment(doctor_name, appt_time, user_name, user_id, user_phone, df)
            return f"Appointment booked for {user_name} with Dr.{doctor_name} at {appt_time}. **Appointment process completed.**"
        else:
            return f"Appointment time {appt_time} is not available for Dr.{doctor_name}. **Appointment process failed.**"
    except Exception as e:
        print(f"Error booking appointment: {e}")
        return "Error booking appointment."

# Function to get appointment details from the user
def get_appointment_details(available_hours_str):
    print("Available appointments: ")
    print(available_hours_str)
    doctor_name = input("Enter doctor's name: ")
    appt_time = input("Enter appointment time (HH:MM): ")
    user_name = input("Enter your name: ")
    user_id = input("Enter your ID: ")
    user_phone = input("Enter your phone number: ")
    return doctor_name, appt_time, user_name, user_id, user_phone

# Function to send an SMS using Twilio
def send_sms(to_phone_number, message_body):
    try:
        message = twilio_client.messages.create(
            body=message_body,
            from_=twilio_phone_number,
            to=to_phone_number
        )
        print(f"SMS sent to {to_phone_number}: {message.sid}")
        return "SMS sent successfully."
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return "Error sending SMS."

# Define tools for LangChain agent
tools = [
    Tool(
        name="Get Available Appointments",
        func=lambda _: get_available_appointments(load_data()),
        description="Gets the available appointment times."
    ),
    Tool(
        name="Book Appointment",
        func=lambda input_str: book_appointment(*input_str.split(", "), load_data()),
        description="Books an appointment. Input: doctor name, appointment time, user name, user id, user phone."
    ),
    Tool(
        name="Get Appointment Details",
        func=lambda _: get_appointment_details(get_available_appointments(load_data())),
        description="Gets the doctor and time for the appointment from the user."
    ),
    Tool(
        name="Send SMS",
        func=lambda input_str: send_sms(*input_str.split(", ")),
        description="Sends an SMS message. Input: phone number, message body."
    )
]

# Initialize LangChain agent with OpenAI LLM
llm = LangChainOpenAI(temperature=0, openai_api_key=api_key)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate"
)

# Main loop for user interaction
while True:
    user_prompt = input("User: ")

    # Handle appointment booking requests
    if "BOOK" == user_prompt:
        available_hours_str = get_available_appointments(load_data())
        doctor_name, appt_time, user_name, user_id, user_phone = get_appointment_details(available_hours_str)
        input_str = f"{doctor_name}, {appt_time}, {user_name}, {user_id}, {user_phone}"
        contact_str = f"{user_phone}, Your appointment with Dr.{doctor_name} at {appt_time} has been booked."
        response = agent.run(f"Book an appointment with {input_str}. If the booking is successful, send SMS with contact_str")

        if "**Appointment process failed.**" in response:
            print("Chatbot: Appointment process has failed. Please choose another time.")
        else:
            print("Chatbot:", response)
    # Handle general medical queries
    else:
        query_vector = model.encode([user_prompt]).astype(np.float32)
        distances, indices = faiss_index.search(query_vector, 5)
        relevant_texts = [texts[idx] for idx in indices[0]]
        context = "\n".join(relevant_texts)
        llm_prompt = f"Context: {context}\nUser Query: {user_prompt}\nAnswer:"
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "developer",
                    "content": "You are a medical assistant chatbot. Your sole purpose is to answer health-related questions. "
                               "Do not respond to any queries outside of the medical domain. When faced with non-medical inquiries, "
                               "prompt that 'I can only assist with health-related matters. How can I help you today?' and offer no further information. "
                               "Ensure your responses are accurate, informative, and based on reliable medical sources. "
                               "Always advise users to consult with a qualified healthcare professional for personalized medical advice. "
                               "Use relevant and appropriate emojis in your responses to make the interaction more engaging and friendly."
                },
                {"role": "user", "content": llm_prompt}
            ]
        )
        response = completion.choices[0].message.content
        print("Chatbot:", response)
