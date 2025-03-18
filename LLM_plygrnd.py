from openai import OpenAI

# API key
api_key = "API_KEY"  

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)  

# Get user input
user_input = str(input("Please enter your health-related question: "))

# Create a completion request for the medical assistant chatbot
completion = client.chat.completions.create(
    model="gpt-4o-mini",  
    messages=[{
        "role": "developer",
        "content": "You are a medical assistant chatbot. Your sole purpose is to answer health-related questions and book appointments for users."
        " Do not respond to any queries outside of the medical domain. When faced with non-medical inquiries, prompt that 'I can only assist with health-related matters.'"
        " and offer no further information. Ensure your responses are accurate, informative, and based on reliable medical sources. "
        "Always advise users to consult with a qualified healthcare professional for personalized medical advice. Use relevant and appropriate emojis in your responses "
        "to make the interaction more engaging and friendly."
    },
    {"role": "user", "content": user_input}
    ]
)

# Output the response from the model
print(completion.choices[0].message.content)
