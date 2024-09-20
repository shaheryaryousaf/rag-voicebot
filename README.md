# 📄 VoiceBot - Chat with Your Documents

### Overview
This application allows users to upload a PDF document and ask questions about the document using voice input in Urdu. The app leverages OpenAI's GPT model to generate responses based on the document's content and also provides audio responses in Urdu. The project uses LangChain for handling document processing, text embeddings, and vector storage, while Qdrant is used as the vector database.

### Key Features
 - 📄 **Document Upload:** Upload a PDF file to the app.
 - 🧠 **Document Understanding:** The app processes the document and stores the text as embeddings in a vector database (Qdrant).
 - 🎙 **Voice Interaction:** Users can ask questions via voice input in Roman Urdu.
 - 📜 **Chat History:** The entire chat history, including both the user's input and the bot's response, is displayed.
 - 🔊 **Audio Output:** The bot’s responses are generated in both text and audio format (using gTTS for text-to-speech).

### Instructions for Local Setup
Follow these steps to run the application on your local machine:

#### 1. Clone the Repository
```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

#### 2. Set Up a Virtual Environment
First, make sure you have Python installed on your machine. Then, create and activate a virtual environment.

For **Windows**:
```
python -m venv env
env\Scripts\activate
```

For **MacOS/Linux**:
```
python3 -m venv env
source env/bin/activate
```

#### 3. Install Dependencies
Install the required dependencies from the ```requirements.txt``` file.
```
pip install -r requirements.txt
```

#### 4. Environment Variables
Before running the application, you need to set up the following environment variables for OpenAI and Qdrant API keys. Create a ```.env``` file in the project directory and add your API keys as follows:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
```

#### 5. Run the Application
Once the environment is set up and all dependencies are installed, you can run the Streamlit app on your local machine:
```
streamlit run app.py
```
This will start the application and launch it in your default web browser at ```http://localhost:8501```.

### Instructions for Using the Application
**Upload PDF File:** On the left side of the screen, you’ll see an option to upload a PDF file. Once uploaded, the app will process the document and store its text in the vector database.

**Ask Questions in Urdu:**
After the document is processed, you can use the voice input to ask questions about the document in Roman Urdu. The bot will provide the response based on the document’s content.

**View and Hear Responses:**
The bot's responses will be displayed as text and audio. You can play the audio directly within the app.

**Chat History:**
The entire chat history will remain visible, including both your input and the bot's responses, along with audio playback for each response.

### Dependencies
The application relies on the following libraries and services:

 - **LangChain:** Used for text processing and document chunking.
 - **OpenAI GPT:** Model used for generating responses based on document context.
 - **gTTS (Google Text-to-Speech):** Converts the text responses into audio format.
 - **Qdrant:** A vector database for storing and querying document embeddings.
 - **Streamlit:** For building the web-based user interface and handling voice input.