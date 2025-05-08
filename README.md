# Healthie - AI-Powered Healthcare Chatbot

## Features
- **AI-Powered Responses**: Provides accurate and context-aware answers to healthcare-related questions.
- **Appointment Booking**: Allows users to check available slots and schedule appointments.
- **Rich Response Formats**: Supports both text-based and tabular responses.
- **Clear Chat History**: Users can reset the conversation anytime.
- **User-Friendly UI**: Built with Streamlit for a seamless and interactive experience.

## Tech Stack
- **Frontend**: Streamlit (Python)
- **Backend**: FastAPI (Python)
- **AI Model**: OpenAI gpt-3.5-turbo
- **Vector Database**: FAISS
- **RAG (Retrieval-Augmented Generation)**: Integrated to enhance response accuracy
- **Tool Calling**: Implemented using LangChain for executing specific tasks
- **Data Handling**: Google Sheets API for appointment management

## How to Use
Click 👉 [here](https://healthie-frontend.streamlit.app/) to access the chatbot.
1. **Start a conversation**: Simply type your healthcare-related question in the chatbox.
2. **Check appointment availability**: Ask something like _"Show me the available hours for the appointment."_
3. **Book an appointment**: Provide details in a message such as "Book an appointment for [patient_name] with ID [patient_id], phone number [patient_number], and email [patient_email] with Dr. [doctor_name] at [time]."
Quick Note: The booking feature is a bit sensitive at the moment. While other sentence formats may work, the given example will definitely work.
4. **Clear history**: Reset the chat by clicking the _Clear history_ button if needed.
5. **Learn more**: Click on _How to Use_ in the UI for guidance on using the chatbot.

