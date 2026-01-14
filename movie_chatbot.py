import os
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.memory.buffer import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Movie Trivia Chatbot")
load_dotenv(dotenv_path="C:\\Users\\shard\\OneDrive\\Documents\\movie_chatbot_project\\movie_chatbot_project\\.env",override = True)
print(os.environ.get("OPENAI_API_KEY"))

# --- PDF Ingestion ---
def initialize_vector_store(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}. Please ensure the file exists.")
    
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting documents into chunks...")
    # Requirements: chunk_size=150, chunk_overlap=20
    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    
    print("Generating embeddings and creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# --- Chatbot Logic ---
def create_chatbot_chain(vectorstore):
    # Initializes the LLM. Ensure OPENAI_API_KEY is present in .env
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    template = """You are MovieBot, a movie trivia expert. Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        memory=memory
    )
    
    return qa_chain

# --- Initialization ---
PDF_FILE_PATH = "C:\\Users\\shard\\OneDrive\\Documents\\movie_chatbot_project\\movie_chatbot_project\\movies_trivia.pdf"

try:
    vector_store = initialize_vector_store(PDF_FILE_PATH)
    qa_chain = create_chatbot_chain(vector_store)
except Exception as e:
    print(f"Error during initialization: {e}")
    qa_chain = None

# --- Gradio UI ---
def chat_response(message, history): 
    print("History before appending:", history)
    # Format the user message properly 
    history.append({"role": "user", "content": message})
    #print("Formatted messages:", messages)
    # Send the query to the chain and get the response 
    response = qa_chain.invoke({"query": message}) 
    # Get the assistant's reply 
    bot_message = response["result"] 
    # Append the user-bot pair to history 
    history.append({"role": "assistant", "content": bot_message}) 
    print("History after appending:", history)
    return "", history

def clear_history():
    if qa_chain:
        qa_chain.memory.clear()
    return None

with gr.Blocks() as demo:
    gr.Markdown("#  Movie Trivia Chatbot")
    chatbot = gr.Chatbot(label="MovieBot")
    msg = gr.Textbox(label="Ask me anything about films:", placeholder="e.g., Who directed The Matrix?")
    
    with gr.Row():
        submit = gr.Button("Submit", variant="primary")
        clear = gr.Button("Clear History")

    def respond(message, chat_history):
        print(f"User Message: {message}") 
        print(f"Chat History: {chat_history}") 
        #chat_history.append({"role": "user", "content": message})
        bot_message, chat_history = chat_response(message, chat_history)
        #chat_history.append(("role": "assistant", "content": bot_message))
        #chat_history.append({"role": "assistant", "content": bot_message})
        print(f"Updated History: {chat_history}")
        return "", chat_history

    submit.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False).then(clear_history)
    
    # Initial greeting
    #chatbot.value = [[None, "Hello, I am MovieBot, your movie trivia expert. Ask me anything about films!"]]
    chatbot.value = [{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Hello, I am MovieBot, your movie trivia expert. Ask me anything about films!"}]}]


if __name__ == "__main__":
    demo.launch()
