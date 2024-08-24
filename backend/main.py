from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
import shutil
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
from config import huggingface_api_token,pinecone_api_key
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
import uuid
app = FastAPI()

DATABASE_SCHEMA = """
-- Users Table
CREATE TABLE IF NOT EXISTS user (
    userid INTEGER PRIMARY KEY,
    surveyid INTEGER
);

-- Surveys Table
CREATE TABLE IF NOT EXISTS survey (
    surveyid INTEGER PRIMARY KEY,
    ques_id INTEGER,
    ques TEXT,
    UNIQUE(surveyid, ques_id)
);

-- Responses Table
CREATE TABLE IF NOT EXISTS responses (
    userid INTEGER,
    surveyid INTEGER,
    quesid INTEGER,
    response TEXT,
    FOREIGN KEY (userid) REFERENCES user(userid),
    FOREIGN KEY (surveyid) REFERENCES survey(surveyid)
);
"""

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment variables
os.environ["HUGGINGFACE_API_TOKEN"] = huggingface_api_token
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# SQLite database connection
DATABASE_URL = "sqlite:///./survey_database.db"

# Directory to store uploaded PDFs
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
pdf_tracker = {}

def init_db():
    conn = sqlite3.connect("survey_database.db")
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user (
        userid INTEGER PRIMARY KEY,
        surveyid INTEGER
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS survey (
        surveyid INTEGER PRIMARY KEY,
        ques_id INTEGER,
        ques TEXT,
        UNIQUE(surveyid, ques_id)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        userid INTEGER,
        surveyid INTEGER,
        quesid INTEGER,
        response TEXT,
        FOREIGN KEY (userid) REFERENCES user(userid),
        FOREIGN KEY (surveyid) REFERENCES survey(surveyid)
    )
    """)

    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM user")
    user_count = cursor.fetchone()[0]

    if user_count == 0:
        # Insert sample data only if the table is empty
        cursor.executemany("INSERT OR IGNORE INTO user (userid, surveyid) VALUES (?, ?)",
                           [(1, 1), (2, 1), (3, 2)])
        cursor.executemany("INSERT OR IGNORE INTO survey (surveyid, ques_id, ques) VALUES (?, ?, ?)",
                           [(1, 1, "How satisfied are you with our service?"),
                            (1, 2, "Would you recommend us to a friend?"),
                            (2, 1, "What features would you like to see in the future?")])
        cursor.executemany("INSERT OR IGNORE INTO responses (userid, surveyid, quesid, response) VALUES (?, ?, ?, ?)",
                           [(1, 1, 1, "Very satisfied"),
                            (1, 1, 2, "Yes, definitely"),
                            (2, 1, 1, "Somewhat satisfied"),
                            (2, 1, 2, "Maybe"),
                            (3, 2, 1, "I would like to see more customization options")])

    conn.commit()
    conn.close()

    print("Database initialized successfully.")

# def get_database_schema():
#     conn = sqlite3.connect("survey_database.db")
#     cursor = conn.cursor()
    
#     # Retrieve the list of tables
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     tables = cursor.fetchall()
    
#     schema = {}
    
#     for table_name in tables:
#         table_name = table_name[0]
#         cursor.execute(f"PRAGMA table_info({table_name});")
#         columns = cursor.fetchall()
#         schema[table_name] = [(col[1], col[2]) for col in columns]  # column name and type
    
#     conn.close()
    
#     return schema

db = SQLDatabase.from_uri(DATABASE_URL)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'testing-pdf-index'

def create_pinecone_index(index_name):
    try:
        # Check if the index already exists
        existing_indexes = pc.list_indexes()
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists.")
            return

        # Create the pc index with the correct dimension
        pc.create_index(
            name=index_name,
            dimension=384,  # Set dimension to match your embeddings
            metric='cosine',
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        if "ALREADY_EXISTS" in str(e):
            print(f"Index '{index_name}' already exists. Continuing with existing index.")
        else:
            print(f"Error creating Pinecone index: {str(e)}")
    
create_pinecone_index(index_name=index_name)

init_db()
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# we can hardcode the db schema and give it to the backend

llm = Ollama(model="llama3")

# Global Pinecone Vector Store
vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)

def generate_sql_query(question: str) -> str:
    
    prompt_template = f"""
    Given the following question and database schema, generate the SQL query that would answer it. 
    Provide only the SQL query without any additional text.

    Database Schema:
    {DATABASE_SCHEMA}

    Question: {{question}}

    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    sql_chain = prompt | llm
    response = sql_chain.invoke({"question": question}).strip()
    
    return response

def query_database(question: str):
    try:
        sql_query = generate_sql_query(question).strip()

        if not sql_query.lower().startswith(("select", "insert", "update", "delete")):
            raise ValueError("Generated query is not a valid SQL statement.")
        
        result = db.run(sql_query)

        if not result:
            return "No results found for your query."

        formatted_result = f"Results from the database query: {result}"
        return formatted_result
    except Exception as e:
        print(f"Error during SQL execution: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"


def get_pdf_retriever():
    return vector_store.as_retriever()
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    all_documents = []
    print(pdf_tracker)
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        # Check if the PDF file already exists in local storage
        if any(file_info["filename"] == file.filename for file_info in pdf_tracker.values()):
            raise HTTPException(status_code=400, detail="File already uploaded locally.")

        # Check if the file_id is already in Pinecone
        index = pc.Index(index_name)
        result = index.query(
            vector=np.zeros(384).tolist(),  # Dummy query vector
            top_k=1,  # Only need to check existence
            include_metadata=True,
            namespace=index_name,
            filter={"file_id": file_id}
        )

        if result and result.get('matches'):
            raise HTTPException(status_code=400, detail="File already uploaded to Pinecone.")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the newly uploaded PDF
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

            if not documents:
                raise HTTPException(status_code=400, detail="No content found in the PDF.")

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)

            if not texts:
                raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

            existing_texts = set()

            for text in texts:
                # Check if the text already exists in Pinecone
                result = index.query(
                    vector=embeddings.embed_documents([text.page_content])[0],
                    top_k=1,
                    include_metadata=True,
                    namespace=index_name
                )

                if result and result.get('matches'):
                    existing_texts.add(text.page_content)

            # Filter out documents that already exist in Pinecone
            new_text_chunks = [
                text.page_content for text in texts if text.page_content not in existing_texts
            ]
            
            pdf_tracker[file_id] = {
                "filename": file.filename,
                "path": file_path,
                "pinecone_namespace": file_id,
                "document_ids": [f"{file_id}_{i}" for i in range(len(new_text_chunks))]
            }
                
            if not new_text_chunks:
                raise HTTPException(status_code=400, detail="No new text to add from the PDF.")

            text_embeddings = embeddings.embed_documents(new_text_chunks)

            if isinstance(text_embeddings, list):
                text_embeddings = np.array(text_embeddings)

            # vectors = [
            #     {
            #         "id": f"{file_id}_{i}",
            #         "values": vec.tolist(),
            #         "metadata": {"text": new_text_chunks[i], "file_id": file_id}
            #     }
            #     for i, vec in enumerate(text_embeddings)
            # ]
            
            # Add all documents to the vector store
            all_documents.extend([
                Document(page_content=new_text_chunks[i], metadata={"file_id": file_id})
                for i in range(len(new_text_chunks))
            ])

            if all_documents:
                # Add all documents to the vector store with file_id
                vector_store.add_documents(documents=all_documents, ids=[f"{file_id}_{i}" for i in range(len(all_documents))])

            return {"message": f"PDF {file.filename} uploaded and processed successfully", "file_id": file_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    question: str
    
@app.post("/chat/")
async def chat(request: ChatRequest):
    question = request.question

    # Get the PDF retriever from the global vector store
    pdf_retriever = get_pdf_retriever()
    
    # Custom prompt template for querying with context
    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Create the RetrievalQA chain
    chain = (
        RunnableParallel({"context": pdf_retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    
    pdf_answer = "No PDF information available."
    if pdf_retriever is not None:
        try:
            pdf_answer = chain.invoke(question)
        except Exception as e:
            pdf_answer = f"Error retrieving PDF information: {str(e)}"
            print(f"Error retrieving PDF information: {str(e)}")
    else:
        print("No PDFs have been processed successfully.")
    
    # Query the database
    db_answer = query_database(question)
    
    # Combine answers from both PDF and database
    final_template = """
    Answer the following question using information from the PDFs and the database. If the information from the PDFs or database is not directly relevant, respond naturally and try to provide a helpful answer based on general knowledge.

    PDF Information: {pdf_answer}
    
    Database Information: {db_answer}
    
    The database includes details about:
    - Users and their associated surveys
    - Survey questions
    - Responses from users to the survey questions
    
    Question: {question}
    
    Combined Answer:
    """

    final_prompt = ChatPromptTemplate.from_template(final_template)
    
    # Generate the final combined answer
    combined_chain = final_prompt | llm
    
    try:
        final_answer = combined_chain.invoke({
            "pdf_answer": pdf_answer,
            "db_answer": db_answer,
            "question": question
        }).strip()
    except Exception as e:
        final_answer = f"Error generating final answer: {str(e)}"
        print(f"Error generating final answer: {str(e)}")
    
    return {"answer": final_answer}

@app.delete("/remove-pdf/{file_id}")
async def remove_pdf(file_id: str):
    if file_id not in pdf_tracker:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Remove from local storage
        file_path = pdf_tracker[file_id]["path"]
        os.remove(file_path)
        
        # Remove all entries related to the file_id from Pinecone
        index = pc.Index(index_name)
        document_ids = pdf_tracker[file_id]["document_ids"]

        # Delete each document ID
        if document_ids:
            index.delete(ids=document_ids)

        # Remove from tracker
        del pdf_tracker[file_id]
        
        return {"message": "PDF removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/')
def sample():
    return 'Running!'

