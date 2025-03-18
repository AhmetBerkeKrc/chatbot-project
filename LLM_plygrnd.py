from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os


# Fetch API key from environment variables for security
api_key = "API_KEY"

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client with the provided API key
client = OpenAI(api_key=api_key)

# Define a Pydantic model for input validation (user query)
class UserInput(BaseModel):
    query: str

# Endpoint for processing medical questions
@app.post("/ask/")
def ask_medical_question(user_input: UserInput):
    try:
        # Call OpenAI API to get a response based on the user's query
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{
                "role": "developer",
                "content": "You are a medical assistant chatbot. Your sole purpose is to answer health-related questions. "
                            "Do not respond to any queries outside of the medical domain. When faced with non-medical inquiries, prompt that 'I can only assist with health-related matters.'"
                            "and offer no further information.  Ensure your responses are accurate, informative, and based on reliable medical sources. "
                            "Always advise users to consult with a qualified healthcare professional for personalized medical advice. Use relevant and appropriate emojis in your responses "
                            "to make the interaction more engaging and friendly."
            },
            {"role": "user", "content": user_input.query}
            ]
        )

        # Return the response from the model
        response = completion.choices[0].message.content
        return {"response": response}

    except Exception as e:
        # In case of error, raise HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

