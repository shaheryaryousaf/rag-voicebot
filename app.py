import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import tempfile
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import speech_to_text
import base64
import os
from dotenv import load_dotenv
load_dotenv()


# Your API keys and URLs (replace with your actual keys)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # To store the chat history
if "data_stored" not in st.session_state:
    # To track if data is stored in vector DB
    st.session_state["data_stored"] = False
if "file_meta" not in st.session_state:
    st.session_state["file_meta"] = {}  # Store file metadata
if "recording_key" not in st.session_state:
    # For unique speech_to_text widget keys
    st.session_state["recording_key"] = 0
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None  # Store the actual uploaded file
if "pdf_processed" not in st.session_state:
    # To track when the PDF is processed and stored
    st.session_state["pdf_processed"] = False


# This function handles the file upload process within a Streamlit app and returns the uploaded file.
def upload_pdf():
    """Handle the file upload and return the uploaded file."""
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        st.session_state["file_meta"] = {
            "file_name": uploaded_file.name,
            "file_size": uploaded_file.size / 1024  # Size in KB
        }
        # Store the actual file
        st.session_state["uploaded_file"] = uploaded_file
    return uploaded_file


# This function processes the uploaded PDF file and stores its content in a Qdrant vector database.
def process_pdf(uploaded_file):
    """Process the uploaded PDF file, create chunks, and store in vector DB."""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load and extract text from the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        document_text = "".join([page.page_content for page in pages])

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Adjust as needed
            chunk_overlap=40  # Adjust as needed
        )
        chunks = text_splitter.create_documents([document_text])

        # Create the embedding model
        embedding_model = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"
        )

        # Send the chunks to the Qdrant vector DB
        if send_to_qdrant(chunks, embedding_model):
            st.session_state["data_stored"] = True
            st.session_state["pdf_processed"] = True
            st.success("Document successfully stored in the vector database.")
        else:
            st.error("Failed to store the document in the vector database.")
    else:
        st.error("Please upload a valid PDF file.")


# This function sends document chunks (with their embeddings) to the Qdrant vector database for storage.
def send_to_qdrant(documents, embedding_model):
    """Send the document chunks to the Qdrant vector database."""
    try:
        qdrant = Qdrant.from_documents(
            documents,
            embedding_model,
            url=QDRANT_URL,
            prefer_grpc=False,
            api_key=QDRANT_API_KEY,
            collection_name="xeven_voicebot",
            force_recreate=True
        )
        return True
    except Exception as ex:
        st.error(f"Failed to store data in the vector DB: {str(ex)}")
        return False


# This function initializes a Qdrant client and returns a vector store object.
def qdrant_client():
    """Initialize Qdrant client and return the vector store."""

    # Initialize the embedding model using OpenAI's text-embedding-ada-002 model.
    # This model will generate 1536-dimensional embeddings from text.
    embedding_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
    )

    # Initialize the Qdrant client with the provided Qdrant URL and API key.
    # This client will interact with the Qdrant vector database where embeddings are stored.
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Create the Qdrant vector store that will be used to store and retrieve document embeddings.
    # The vector store is associated with the specified collection "xeven_voicebot".
    qdrant_store = Qdrant(
        client=qdrant_client,                # The Qdrant client instance
        # The name of the collection to store embeddings
        collection_name="xeven_voicebot",
        embeddings=embedding_model           # The embedding model used to generate vectors
    )

    # Return the Qdrant vector store for use in storing and querying embeddings.
    return qdrant_store


# This function captures audio input from the user's microphone and converts it to text using a speech-to-text engine.
def process_audio_input():
    """Capture audio from mic and convert it to text using speech-to-text."""

    # Generate a unique key for this recording session based on the recording key in session state
    unique_key = f"STT-{st.session_state['recording_key']}"

    # Capture audio input from the mic and convert it to text (using Urdu as the language)
    transcribed_text = speech_to_text(
        language='ur',               # Set language to Urdu for speech recognition
        # Ensure the widget adapts to the container width in the UI
        use_container_width=True,
        just_once=True,              # Capture audio only once for simplicity
        key=unique_key               # Unique key for this specific recording session
    )

    # Return the transcribed text for further use
    return transcribed_text


# Converts text to speech in the specified language using gTTS and returns the audio data.
def text_to_speech(text, lang='ur'):
    """Convert the assistant's response to audio in Urdu using gTTS."""

    # Initialize gTTS with the given text and language
    tts = gTTS(text=text, lang=lang, tld='com.au')

    # Create an in-memory buffer for audio data
    audio_data = BytesIO()

    # Write the generated audio to the buffer
    tts.write_to_fp(audio_data)

    # Reset buffer to the start
    audio_data.seek(0)

    # Return the audio data
    return audio_data


# Sends a query to the Qdrant vector store and generates a response using a GPT model
def qa_ret(qdrant_store, input_query):
    """Retrieve relevant documents and generate a response from the AI model."""
    try:
        # Define the template for generating responses, including context and the user's question.
        template = """
        You are a helpful and dedicated female assistant. Your primary role is to assist the user by providing accurate
        and thoughtful answers based on the given context. If the user asks any questions related to the provided
        information, respond in a courteous and professional manner.
        **Note:** Always provide your responses in Roman Urdu (Hinglish) to maintain the preferred language format.
        {context}
        **Question:** {question}
        """
        # Create a chat prompt template based on the defined template
        prompt = ChatPromptTemplate.from_template(template)

        # Configure Qdrant to retrieve relevant documents based on similarity search
        retriever = qdrant_store.as_retriever(
            # Retrieve top 4 relevant results
            search_type="similarity", search_kwargs={"k": 4}
        )

        # Set up parallel execution to retrieve context and pass it with the question
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )

        # Initialize the GPT model (gpt-4o-mini) with a moderate creativity level
        model = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )

        # Define an output parser to handle the response formatting
        output_parser = StrOutputParser()

        # Chain the setup, prompt, model, and output parsing into one pipeline
        rag_chain = setup_and_retrieval | prompt | model | output_parser

        # Generate and return the response using the input query
        response = rag_chain.invoke(input_query)
        return response

    # Handle any exceptions and return an error message
    except Exception as ex:
        return f"Error: {str(ex)}"


# Displays the chat history, including user inputs and assistant responses.
# 1. Iterates over the chat history stored in session state.
# 2. For each message, displays the user input and assistant's response.
# 3. Also plays the assistant's response in audio format (if available).
def display_chat_history():
    """Display the chat history."""
    
    # Iterate over each chat entry in the chat history stored in session state
    for chat in st.session_state["chat_history"]:
        
        # Display the user's input in a chat message format
        with st.chat_message("user"):
            st.write(f"**You:** {chat['user_input']}")  # Show user input
        
        # Display the assistant's response in a chat message format
        with st.chat_message("assistant"):
            st.write(f"**Bot:** {chat['bot_response']}")  # Show assistant's response
            st.audio(chat["bot_audio"], format="audio/mp3")  # Play the assistant's audio response
        
        # Display a horizontal divider after each chat entry
        st.write("---")



# Function to display PDF preview
def display_pdf_preview():
    """Display a preview of the uploaded PDF file in the right column."""
    if st.session_state["pdf_processed"]:
        uploaded_file = st.session_state["uploaded_file"]
        if uploaded_file:
            # Process and display PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Read the file as base64
            with open(temp_file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            # Embed PDF preview using iframe
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)


# Main App Flow
def main():
    # Set up the page configuration (title, layout, etc.)
    st.set_page_config(
        page_title="Voicebot - Chat with Documents",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main Title
    st.title("📄 Voicebot - Chat with Your Documents")

    # Subheader
    st.subheader(
        "Upload a document, ask questions, and get responses in both text and audio form in Urdu.")
    st.write("This app allows you to interact with documents in Urdu. It uses advanced NLP to process documents, allowing you to ask questions and get detailed answers in Urdu, both in text and audio format.")

    # Sidebar instructions and about section
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    **How to Use:**
     1. Upload a PDF or document file using the upload button below.
    2. Once processed, ask your question in Urdu by typing or using voice input.
    3. Get responses in text and audio form (in Urdu).
    4. You can ask multiple questions sequentially, and your previous responses will remain visible.
    
    **Notes:**
    - The answers are based on the document you uploaded.
    - You can ask multiple questions one after another.
    - Your chat history will be visible throughout the session.
    """)

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Left column: Upload and Chat
        st.header("Upload and Chat")

        # Step 1: Upload the PDF file
        uploaded_file = upload_pdf()

        # Step 2: Process the PDF when 'Process' button is clicked
        if uploaded_file is not None and not st.session_state["pdf_processed"]:
            if st.button("Process"):
                process_pdf(uploaded_file)

        # Step 3: Show the audio input if the document is stored in vector DB
        if st.session_state["data_stored"]:
            # Record audio input
            st.write("Ask Your Question in Urdu")
            audio_input = process_audio_input()

            if audio_input:
                # Increment recording key for unique widget IDs
                st.session_state["recording_key"] += 1

                # Retrieve the Qdrant store for retrieval
                qdrant_store = qdrant_client()

                # Generate response from AI model
                bot_response = qa_ret(qdrant_store, audio_input)

                # Convert bot's response to audio (in Urdu)
                audio_response = text_to_speech(bot_response, lang='ur')

                # Append to chat history with file metadata
                st.session_state["chat_history"].append({
                    "user_input": audio_input,
                    "bot_response": bot_response,
                    "bot_audio": audio_response,
                    "file_meta": st.session_state["file_meta"]
                })

                # Display the chat history
                display_chat_history()

                # Place the record button under the latest answer
                if st.button("Ask Another Question"):
                    st.experimental_rerun()

    with col2:
        # Right column: Preview of the uploaded file
        st.write("Uploaded File Preview")

        # Only display the file preview if there is at least one message in chat_history
        if len(st.session_state["chat_history"]) > 0:
            display_pdf_preview()


# Run the app
if __name__ == "__main__":
    main()
