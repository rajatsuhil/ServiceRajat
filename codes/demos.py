import os
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def build_vectorstore(pdf_path: str, embeddings, index_dir: str = "faiss_index_react"):
    """
    Load PDF, split into chunks, generate embeddings, and save FAISS index.
    """
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_dir)
    return vectorstore


def load_vectorstore(embeddings, index_dir: str = "faiss_index_react"):
    """
    Load an existing FAISS index from disk.
    """
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def initialize_llm(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    """
    Load tokenizer and model into a HuggingFace pipeline wrapped by LangChain.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=llm_pipe)


def main():
    # 1) Ensure HF Token
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face API token: ").strip()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    # 2) Initialize LLM and Embeddings
    llm = initialize_llm()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3) Build or Load FAISS index
    index_dir = "faiss_index_react"
    if not os.path.exists(index_dir):
        pdf_path = input("Enter the path to your PDF file: ").strip()
        print("Building FAISS index from PDF...")
        vectorstore = build_vectorstore(pdf_path, embeddings, index_dir)
    else:
        print("Loading existing FAISS index...")
        vectorstore = load_vectorstore(embeddings, index_dir)

    # 4) Setup Retrieval-Augmented Generation chain
    print("Setting up retrieval chain...")
    retrieval_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_chain = create_stuff_documents_chain(llm, retrieval_chat_prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_chain)

    # 5) Interactive Q&A loop
    print("\nPDF QA CLI ready! Type your question or 'exit' to quit.")
    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        result = retrieval_chain.invoke({"input": user_query})
        print(f"\nAnswer:\n{result['answer']}\n")


if __name__ == "__main__":
    main()
