from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from datasets import load_dataset

# API KEY
api_key = "API_KEY"

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Load Sentence Transformer model for embedding generation
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load FAISS index for efficient similarity search
faiss_index = faiss.read_index("medical_faiss.index")

# Load dataset and extract text data
dataset = load_dataset("codexist/medical_data")
texts = [row["data"] for row in dataset["train"]]

# Initialize FastAPI application
app = FastAPI()

# Define a model to handle user input
class UserInput(BaseModel):
    user_prompt: str

# Define the main route for chat interactions
@app.post("/chat")
async def chat_with_doctor(user_input: UserInput):
    # Convert user query to vector using the SentenceTransformer model
    query_vector = model.encode([user_input.user_prompt]).astype(np.float32)

    # Perform a search in the FAISS index to find the closest matching texts
    distances, indices = faiss_index.search(query_vector, 5)  # Retrieve top 5 most relevant results

    # Extract the relevant texts based on the indices from the search
    relevant_texts = [texts[idx] for idx in indices[0]]

    # Combine the relevant texts into a single context string to send to the LLM
    context = "\n".join(relevant_texts)
    llm_prompt = f"Context: {context}\nUser Query: {user_input.user_prompt}\nAnswer:"

    # Send the query to OpenAI API to get a response from the medical assistant model
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

    # Return the generated response and the relevant context (RAG data)
    return {
        "response": completion.choices[0].message.content,
        "rag_context": relevant_texts
    }
