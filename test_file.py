import openai
import os

# Set the API key
openai.api_key = os.getenv('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')

# Ensure the API key is set
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

try:
    # Use the new API for embeddings
    response = openai.Embedding.create(
        input=["Test"],  # Input should be a list of strings
        model="text-embedding-ada-002"  # Ensure this is a valid model
    )
    print("Embedding response:", response)
except openai.error.APIConnectionError as e:
    print("APIConnectionError:", e)
except Exception as e:
    print("General error:", e)