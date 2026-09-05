from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


# Load environment variables
load_dotenv()


# Pinecone API Key
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# Download/load Hugging Face embeddings
embeddings = download_hugging_face_embeddings()


# Pinecone index
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# ============================================================
# Ollama Local LLM
# ============================================================

chatModel = ChatOpenAI(
    model="llama3.2:1b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# Question Answer Chain
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)


# RAG Chain
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# ============================================================
# Flask Routes
# ============================================================

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    print("Question:", msg)

    response = rag_chain.invoke(
        {
            "input": msg
        }
    )

    print("Response:", response["answer"])

    return str(response["answer"])


# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )