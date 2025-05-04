# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter
# # from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore

# load_dotenv()

# if _name_ == '_main_':
#     print("Ingesting...")
#     # below ecoding need to be specid=fied by you ither wise error
#     loader = TextLoader(r"D:\All_my_data\Langchain\vector_db\mediumblog1.txt", encoding="utf-8")
#     document = loader.load()

#     print("splitting...")
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(document)
#     print(f"created {len(texts)} chunks")

#     embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

#     print("ingesting...")
#     PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
#     print("finish")


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

def initialize_llama_embeddings():
    """Initialize Llama-based embeddings from Hugging Face"""
    print("Initializing Llama-based embeddings...")
    
    # Use a Hugging Face sentence transformer model compatible with Llama architecture
    # You can replace this with a specific Llama embedding model if you have one
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight and efficient
    
    # For true Llama embeddings, you might use something like:
    # embedding_model_name = "llama-embeddings/llama-e5-base-768"  # If you have access
    
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return embeddings

def initialize_llama_llm():
    """Initialize Llama LLM from Hugging Face (optional)"""
    print("Initializing Llama LLM...")
    
    # Check if HUGGINGFACEHUB_API_TOKEN is set
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        hf_token = input("Enter your Hugging Face API token (or set it in .env file): ")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    
    # Use a Llama model - adjust based on your access/requirements
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Requires HF access to this model
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        llm_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )
        
        llm = HuggingFacePipeline(pipeline=llm_pipe)
        return llm
    except Exception as e:
        print(f"Error initializing Llama LLM: {e}")
        print("Continuing without LLM initialization (only embeddings will be used)")
        return None

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "INDEX_NAME"]
    missing_vars = [var for var in required_vars if os.environ.get(var) is None]
    
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or provide them now:")
        
        for var in missing_vars:
            os.environ[var] = input(f"Enter your {var}: ")

if __name__ == "__main__":
    print("Starting ingestion process...")
    
    # Check environment variables
    check_environment()
    
    # Initialize Llama embeddings
    embeddings = initialize_llama_embeddings()
    
    # Initialize Llama LLM (optional - if you need it for other operations)
    llm = initialize_llama_llm()
    
    print("Loading document...")
    # Load the document with specified encoding
    loader = TextLoader(r"D:\All_my_data\Langchain\vector_db\mediumblog1.txt", encoding="utf-8")
    document = loader.load()
    
    print("Splitting into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")
    
    print("Ingesting into Pinecone...")
    # Create or update Pinecone vector store with Llama embeddings
    PineconeVectorStore.from_documents(
        texts, 
        embeddings, 
        index_name=os.environ['INDEX_NAME']
    )
    
    print("Ingestion complete!")