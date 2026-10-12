import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# 1. Load variables from the .env file into the system environment
load_dotenv()

# 2. Retrieve the key from the environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERROR: Could not find 'GEMINI_API_KEY' in your .env file.")
    print("Please double-check your .env file layout and file path.")
    exit(1)

try:
    # 3. Initialize the client (it automatically picks up GEMINI_API_KEY from environment)
    client = genai.Client()
    
    # 4. Attempt a simple request
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Hi! Are you online and working?',
    )
    
    print("✅ SUCCESS: Your .env file is loaded and the API key works perfectly!")
    print(f"Gemini's Response: {response.text}")

except APIError as e:
    print("❌ ERROR: The API accepted the request but returned an error.")
    print(f"Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
