import streamlit as st
import time
import pandas as pd 
import requests

CHATBOT_URL = "http://localhost:8000/chat"


# Check if the session has a messages key, if not, initialize it with a welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello, I am Healthie, your healthcare assistant 🤖. How can I assist you today?"}]

# Initialize session state variable for showing info
if "show_info" not in st.session_state:
    st.session_state.show_info = False

# Initialize session state variable for avatars
if "avatars" not in st.session_state:
    st.session_state.avatars = "" 

# Define layout with three columns for buttons
with st.container():
    col1, col2, col3 = st.columns(3)

    # Toggle "How to Use" information when button is clicked
    if col1.button("How to Use"):
        st.session_state.show_info = not st.session_state.show_info

    # Clear chat history when the button is pressed
    if col2.button("Clear history", key="clear_button"):
        st.session_state.messages = []
        success_message = st.success('The chat history has been deleted.', icon="✅")
        time.sleep(2)
        success_message.empty()
    
    # Link to the project repository
    col3.link_button("Project Repo", "https://github.com/AhmetBerkeKrc/chatbot-project")

# Display usage instructions if the user toggles "How to Use"
if st.session_state.show_info:
    st.info(""" 
    Healthie is your healthcare assistant. You can ask any healthcare-related questions, and it will provide you with answers. Additionally, 
    you can easily book a doctor’s appointment through the chatbot.
    To view available appointment slots, simply type something like: 'Show me the available hours for the appointment.' Once you’ve selected 
    a time slot, you can confirm your booking by typing like: 'Book an appointment for [patient_name] with ID [patient_id], phone number 
    [patient_number], and email [patient_email] with Dr. [doctor_name] at [time]'. 
    The chatbot will take care of the rest!
    """)

# Display the chatbot header
st.header("_Healthie_", divider="gray")

# Loop through session messages and display them in the chat
for message in st.session_state.messages:
    #role_avatar(message)
    with st.chat_message(message["role"]):
        if type(message["content"]) == str:
            st.markdown(message["content"])
        else:
            st.table(message["content"].set_index(message["content"].columns[0]))

# Capture user input and send it to the chatbot
if prompt := st.chat_input("Drop your message here 👇"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send prompt to the chatbot backend
    response = requests.post(CHATBOT_URL, json={"user_prompt": prompt})

    # Display assistant's response
    with st.chat_message("assistant"):
        time.sleep(1)
        # If the response is a list, display it as a table
        if type(response.json()["result"]) == list:
            df = pd.DataFrame(response.json()["result"])
            st.table(df.reset_index(drop=True).set_index(df.columns[0]))
            result_table = pd.DataFrame(response.json()["result"])
            st.session_state.messages.append({"role": "assistant", "content": result_table})
        else:
            # If it's a string, simply display it
            st.write(response.json()["result"])
            response = response.json()["result"]
            st.session_state.messages.append({"role": "assistant", "content": response})

        
        

    