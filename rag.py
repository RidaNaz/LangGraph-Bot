import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
os.environ ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Load LLM 
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash", streaming=True)

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = PineconeVectorStore.from_existing_index(index_name="demo", embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

def get_rag_response(user_query):
    """Retrieves relevant context and generates a response using Openai"""
    retrieved_docs = retriever.invoke(user_query)

    # Formatting context
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # Generate AI response
    prompt = f"Use the following restaurant data to answer: {context} \n\nCustomer: {user_query} \nAI:"
    response = llm.invoke(prompt)

    return response