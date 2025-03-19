# Import necessary libraries
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# Function to load appointment data from Google Sheets using credentials
def load_appointment_list():
    # Define Google Sheets API scopes and authorize using service account credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file("appointment_credential.json", scopes=scopes)
    client = gspread.authorize(creds)
    
    # Access the Google Sheet by its ID and select the desired worksheet
    sheet_id = "11dSw6vLfkGpqeg3nApvGo4bM7Mwp9eQ_EOKog7Q76-M"
    sheet = client.open_by_key(SHEET_ID)
    worksheet = sheet.worksheet("app_sheet")
    
    # Retrieve all data from the worksheet
    values = worksheet.get_all_values()
    columns = values[0]
    
    # Return the data as a Pandas DataFrame for further processing
    return pd.DataFrame(values[1:], columns=columns), worksheet

# Tool to fetch available appointment hours based on a user's request
@tool
def available_hours(listing_request: str) -> str:
    """
    Fetches and returns available appointment hours from a Google Sheets schedule.

    Args:
        listing_request (str): User's request to see available hours.

    Returns:
        str: A list of available appointment slots.
    """
    try:
        # Load the appointment list and extract available slots
        df, worksheet = load_appointment_list()
        available_hours = df[df["Status"] == "Available"].reset_index(drop=True)

        # Return the formatted available hours for the user
        return available_hours[["Doctor Name", "Date", "Time"]].to_string()
    
    except Exception as e:
        # In case of error, return the error message
        return f"Failed to retrieve the data: {e}"

# Tool to book an appointment based on user's details and selected time
@tool
def book_appointment(doctor_name: str, time: str, patient_name: str, patient_id: str, patient_number: str, patient_email: str) -> str:
    """
    Books an appointment for the patient with the given details.

    Args:
        doctor_name (str): Doctor's name.
        time (str): Appointment time (HH:MM format).
        patient_name (str): Patient's full name.
        patient_id (str): Patient's ID number.
        patient_number (str): Patient's phone number.
        patient_email (str): Patient's email address.

    Returns:
        str: Result of the booking process.
    """
    try:
        # Load the appointment list and find the requested slot
        df, worksheet = load_appointment_list()
        mask = (df["Doctor Name"] == doctor_name) & (df["Time"] == time) & (df["Status"] == "Available")

        # If no available slots match, return a failure message
        if not mask.any():
            return f"Appointment booking failed: No available slots for Dr. {doctor_name} at {time}."

        # Update the appointment details in the dataframe
        df.loc[mask, "Patient Full Name"] = patient_name
        df.loc[mask, "Patient ID Number"] = patient_id
        df.loc[mask, "Patient Phone Number"] = patient_number
        df.loc[mask, "Patient Email"] = patient_email
        df.loc[mask, "Status"] = "Booked"

        # Update the Google Sheets worksheet with the new data
        updated_values = [df.columns.tolist()] + df.values.tolist()
        worksheet.update(values=updated_values, range_name="A1")

        # Return success message
        return f"Appointment successfully booked for {patient_name} with Dr. {doctor_name} at {time}."

    except Exception as e:
        # If any error occurs during booking, return the error message
        return f"Booking failed: {e}"

# Define the mapping of tool names to function calls
tool_mapping = {"available_hours": available_hours, "book_appointment": book_appointment}

# Initialize the ChatGPT model with OpenAI API
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.6,
    openai_api_key="API_KEY"
)

# Bind the tools to the model
llm_with_tools = llm.bind_tools([available_hours, book_appointment])

# Create a template for generating the chatbot's responses
template = PromptTemplate(
    input_variables=['context', 'user_input'],
    template=""" 
    You are a professional medical chatbot designed to assist users with health-related queries and providing the current available hours in the schedule.
    
    ### Guidelines:
    - Provide **clear, concise, and medically accurate** responses.
    - Use the provided **context** as the base for your answer.
    - **Do not diagnose, prescribe medications, or suggest treatments** beyond general health advice.
    - **Health is crucial**, so avoid absolute statements; instead, **encourage consulting a healthcare professional** when needed.
    - If the user greets you, respond appropriately with a greeting.
    - If the user asks about **non-medical topics**, respond strictly with: "I can only assist you with your health-related questions, checking available hours, and booking an appointment."
    - If the user asks for the **current available hours in the schedule**, retrieve and provide the available slots from the Google Sheets schedule.
    - If the user wants to **book an appointment**, update columns accordingly with the given arguments from Google Sheets schedule and update it on Google Sheets.

    ### Context:
    {context}

    ### User Input:
    {user_input}
    """
)

# Create a chain for generating responses based on the user input and context
model_chain = template | llm_with_tools

# Load the Sentence Transformer model for semantic search
st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load the FAISS index for fast similarity search
faiss_index = faiss.read_index("medical_faiss.index")

# Load the dataset and extract the texts
dataset = load_dataset("codexist/medical_data")
texts = [row["data"] for row in dataset["train"]]

from langchain_core.messages import ToolMessage

# Loop to interact with the user
while True:
    user_input = str(input("User: "))  # Get user input
    query_vector = st_model.encode([user_input]).astype(np.float32)  # Convert input to vector
    distances, indices = faiss_index.search(query_vector, 5)  # Search for relevant texts
    relevant_texts = [texts[idx] for idx in indices[0]]  # Extract the relevant texts
    context = "\n".join(relevant_texts)  # Combine relevant texts for context
    
    # Generate the response based on context and user input
    response = model_chain.invoke({'context': context, 'user_input': user_input})
    
    # Process any tool calls in the response
    y = None
    for tool_call in response.tool_calls:
        tool = tool_mapping[tool_call["name"].lower()]  # Find the corresponding tool
        tool_output = tool.invoke(tool_call["args"])  # Invoke the tool with the arguments
        X = ToolMessage(tool_output, tool_call_id = tool_call["id"])  # Create a tool message
        y = X.content  # Capture the tool's output
    
    # If no tool output, set response to empty
    if y is None:
        y=""
    
    # Print the chatbot's final response
    print(f"Chatbot: \n{response.content + y}")
