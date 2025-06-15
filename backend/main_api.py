# main_api.py

import os
import glob
import re
import PyPDF2
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from pathlib import Path

# Debug: Print current working directory and .env file status
print(f"Current working directory: {os.getcwd()}")

# Try to load .env from multiple locations
env_paths = [
    Path(os.getcwd()) / '.env',  # Current directory
    Path(os.getcwd()).parent / '.env',  # Parent directory
    Path(__file__).parent / '.env',  # Same directory as this file
    Path(__file__).parent.parent / '.env',  # Parent of this file's directory
]

for env_path in env_paths:
    print(f"Checking for .env at: {env_path}")
    if env_path.exists():
        print(f"Found .env at: {env_path}")
        load_dotenv(env_path)
        break
else:
    print("No .env file found in any of the checked locations")

# Debug: Print environment variables
api_key = os.getenv('OPENAI_API_KEY')
print(f"OPENAI_API_KEY from env: {'Found' if api_key else 'Not found'}")

# --- FastAPI Imports ---
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# --- LangChain Imports ---
# For text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Core components for schemas, prompts, and documents
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Specific integration packages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Graph component
from langgraph.graph import StateGraph, END

# --- Environment and API Key Configuration ---
# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OpenAI API key not found. Please create a .env file in the project root "
        "with your OPENAI_API_KEY or set it as an environment variable."
    )
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Global Configuration ---
# Directory where your legal PDFs are stored
PDF_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algerian_legal_pdfs")
# Directory where the vector database will be persisted
DB_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algerian_laws_db")
DB_COLLECTION_NAME = "algerian_legal_docs"

# ==============================================================================
# 1. VECTOR DATABASE BUILDER (Same as before)
# ==============================================================================
class DocumentVectorDBBuilder:
    """Builder for a generic document vector database."""
    def __init__(self, pdf_dir: str, db_persist_directory: str, collection_name: str):
        self.pdf_dir = pdf_dir
        self.db_persist_directory = db_persist_directory
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        print(f"Extracting text from {os.path.basename(pdf_path)}...")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
            text = re.sub(r'\s+', ' ', text)
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def process_pdfs(self) -> List[Document]:
        all_chunks = []
        pdf_files = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_dir}")

        print(f"Found {len(pdf_files)} PDF files to process.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            metadata = {"source": filename}
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                chunks = text_splitter.create_documents(texts=[text], metadatas=[metadata])
                all_chunks.extend(chunks)
        return all_chunks

    def build_vector_db(self):
        documents = self.process_pdfs()
        if not documents:
            print("No documents to process.")
            return

        print(f"Building vector database with {len(documents)} chunks...")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_persist_directory,
            collection_name=self.collection_name
        )
        vectordb.persist()
        print(f"Vector database built and saved to {self.db_persist_directory}")

# ==============================================================================
# 2. MULTI-AGENT SYSTEM (Modified for async operations)
# ==============================================================================

# --- State Schema ---
class AnalysisState(TypedDict):
    user_query: str
    retrieved_context: str
    synthesized_answer: str
    final_response: str
    messages: Annotated[Sequence[Any], "thread"]

# --- Initialize LLM ---
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)

# --- Agent 1: Retrieval ---
async def retrieval_agent(state: AnalysisState) -> AnalysisState:
    print("--- Agent: Retrieving Context ---")
    user_query = state['user_query']
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        collection_name=DB_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PERSIST_DIRECTORY
    )
    results = await vector_db.asimilarity_search(user_query, k=5)
    context = "\n\n".join([f"--- Context from {doc.metadata.get('source', 'Unknown')} ---\n{doc.page_content}" for doc in results])
    state['retrieved_context'] = context
    state['messages'].append(AIMessage(content=f"Retrieved {len(results)} relevant document sections."))
    return state

# --- Agent 2: Synthesis ---
async def synthesis_agent(state: AnalysisState) -> AnalysisState:
    print("--- Agent: Synthesizing Answer ---")
    system_prompt = """You are an expert analyst of Algerian legal and administrative texts. Your task is to answer the user's query based *only* on the provided context from official documents. Be precise, factual, and respond in French."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: \"{state['user_query']}\"\n\nRetrieved Context:\n{state['retrieved_context']}\n\nBased on the context above, provide a direct and factual answer.")
    ])
    synthesis_chain = prompt | llm | StrOutputParser()
    synthesized_answer = await synthesis_chain.ainvoke({})
    state['synthesized_answer'] = synthesized_answer
    state['messages'].append(AIMessage(content="Synthesized a preliminary answer."))
    return state

# --- Agent 3: Report Generation ---
async def report_agent(state: AnalysisState) -> AnalysisState:
    print("--- Agent: Generating Final Report ---")
    system_prompt = """You are an assistant responsible for generating clear and professional reports in French. Your task is to take the synthesized analysis and format it into a final response using markdown for clarity."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Original User Query: \"{state['user_query']}\"\n\nSynthesized Answer to be Formatted:\n{state['synthesized_answer']}\n\nPlease format this information into a final, comprehensive report.")
    ])
    report_chain = prompt | llm | StrOutputParser()
    final_response = await report_chain.ainvoke({})
    state['final_response'] = final_response
    state['messages'].append(AIMessage(content="Final report generated."))
    return state

# --- Build the Graph ---
def build_document_analysis_graph():
    workflow = StateGraph(AnalysisState)
    workflow.add_node("retrieve_context", retrieval_agent)
    workflow.add_node("synthesize_answer", synthesis_agent)
    workflow.add_node("generate_report", report_agent)
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "synthesize_answer")
    workflow.add_edge("synthesize_answer", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()

# Instantiate the graph globally so it's ready for requests
analysis_graph = build_document_analysis_graph()


# ==============================================================================
# 3. FASTAPI APPLICATION
# ==============================================================================

app = FastAPI(
    title="Algerian Legal Document Analysis API",
    description="An API to query and analyze Algerian legal documents.",
    version="1.0.0"
)

# --- Pydantic Models for Request/Response ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    report: str

class BuildResponse(BaseModel):
    status: str
    message: str

# --- API Endpoints ---
@app.post("/build-database", response_model=BuildResponse, tags=["Database"])
async def build_database_endpoint():
    """
    Triggers the process to build the vector database from the PDFs
    in the configured directory. This should be run once before querying.
    """
    if not os.path.exists(PDF_DIRECTORY) or not glob.glob(os.path.join(PDF_DIRECTORY, "*.pdf")):
        raise HTTPException(
            status_code=404,
            detail=f"PDF directory '{PDF_DIRECTORY}' not found or is empty. Please create it and add PDF files."
        )
    try:
        builder = DocumentVectorDBBuilder(
            pdf_dir=PDF_DIRECTORY,
            db_persist_directory=DB_PERSIST_DIRECTORY,
            collection_name=DB_COLLECTION_NAME
        )
        builder.build_vector_db()
        return {"status": "success", "message": "Database built successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build database: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Analysis"])
async def analyze_query_endpoint(request: QueryRequest = Body(...)):
    """
    Accepts a query, runs it through the analysis workflow, and returns
    the final report.
    """
    if not os.path.exists(DB_PERSIST_DIRECTORY):
        raise HTTPException(
            status_code=400,
            detail="Database not found. Please run the `/build-database` endpoint first."
        )

    initial_state = {
        "user_query": request.query,
        "messages": [],
    }

    try:
        result_state = await analysis_graph.ainvoke(initial_state)
        final_report = result_state.get("final_response")
        if not final_report:
            raise HTTPException(status_code=500, detail="Failed to generate a final report from the workflow.")
        return {"report": final_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Algerian Legal Document Analysis API. See /docs for endpoints."}

# To run the app, use the command: uvicorn main_api:app --reload