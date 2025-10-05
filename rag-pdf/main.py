import base64
from io import BytesIO
import json
import os
import uuid
import fitz   
from pdf2image import convert_from_path  
from PIL import Image  
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import lancedb
import pyarrow as pa
import pandas as pd
import ragas as rg
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from datasets import Dataset

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

bg_embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")  
sbert_embedder = SentenceTransformer("all-MiniLM-L6-v2")
#gemma_embedder = SentenceTransformer("gemma2-large-it")

embedders = [bg_embedder, sbert_embedder] #,gemma_embedder]

uri = "data/rag-database"
db = lancedb.connect(uri)

combo_parameters =[
    {"chunk_size": 300, "overlap": 200, "embedder": bg_embedder, "table_name": "rag-database-bg-300-200", "embedder_name": "bg"},
    {"chunk_size": 512, "overlap": 128, "embedder": bg_embedder, "table_name": "rag-database-bg-512-128", "embedder_name": "bg"},
    {"chunk_size": 256, "overlap": 64, "embedder": bg_embedder, "table_name": "rag-database-bg-256-64", "embedder_name": "bg"},
    {"chunk_size": 256, "overlap": 128, "embedder": bg_embedder, "table_name": "rag-database-bg-256-128", "embedder_name": "bg"},

    {"chunk_size": 200, "overlap": 30, "embedder": sbert_embedder, "table_name": "rag-database-sbert-200-30", "embedder_name": "sbert"   },
    {"chunk_size": 512, "overlap": 128, "embedder": sbert_embedder, "table_name": "rag-database-sbert-512-128", "embedder_name": "sbert"},
    {"chunk_size": 256, "overlap": 64, "embedder": sbert_embedder, "table_name": "rag-database-sbert-256-64", "embedder_name": "sbert"},
    {"chunk_size": 256, "overlap": 128, "embedder": sbert_embedder, "table_name": "rag-database-sbert-256-128", "embedder_name": "sbert"},

    # add more models here and combo parameters

    #{"chunk_size": 200, "overlap": 30, "embedder": gemma_embedder, "table_name": "rag-database-gemma-200-30"},
    #{"chunk_size": 512, "overlap": 128, "embedder": gemma_embedder, "table_name": "rag-database-gemma-512-128"},
    #{"chunk_size": 256, "overlap": 64, "embedder": gemma_embedder, "table_name": "rag-database-gemma-256-64"},
    #{"chunk_size": 256, "overlap": 128, "embedder": gemma_embedder, "table_name": "rag-database-gemma-256-128"}
]

def init_database_schema():
    # Create schema with fixed-size list for vectors (required by LanceDB for vector search)
    for parameter in combo_parameters:
        sample_embedding = parameter["embedder"].encode(["sample"], normalize_embeddings=True)[0]
        embedding_dim = len(sample_embedding)
        print(f"Creating schema for {parameter['table_name']} with embedding dimension: {embedding_dim}")
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=embedding_dim)),  # Fixed-size list required by LanceDB
            pa.field("text", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("chunk_id", pa.string())
        ])
        # check if the table already exists in the database
        if parameter["table_name"] in db.table_names():
            print(f"Table {parameter['table_name']} already exists")
            continue
        db.create_table(parameter["table_name"], schema=schema)

    print("Database schema created successfully")

def clear_existing_tables():
    """Clear existing tables that might have incorrect schema"""
    existing_tables = db.table_names()
    for table_name in existing_tables:
        if any(param["table_name"] == table_name for param in combo_parameters):
            print(f"Dropping existing table: {table_name}")
            db.drop_table(table_name)
    print("Cleared existing tables")

# opens an image based pdf document
def read_from_image_pdf(pdf_path):
    doc = convert_from_path(pdf_path)
    return doc

def read_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return doc

# converts to base64
def convert_to_base64(img):
    buffer = BytesIO()                                 
    img.save(buffer, format="PNG")              
    buffer.seek(0) 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Chunks it with different parameters
def chunk_image_pdf_page(doc, page, chunk_size, overlap):
    text = pytesseract.image_to_string(doc[page])
    return chunk_text(text, chunk_size, overlap)

def chunk_pdf_page(doc, page, chunk_size, overlap):  
    return chunk_text(doc[page].get_text(), chunk_size, overlap)

def chunk_text(text, chunk_size, overlap):
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
    chunks = []
    for text_chunk in text_chunks:
        chunks.append({
            "text": text_chunk,
            "chunk_id": str(uuid.uuid4())
        })
        #print(f"Chunk: {text_chunk}")
    return chunks

# Embeds the chunks into a vector store
def embed_chunks(chunks, embedder):
    chunk_vectors = embedder.encode(chunks, normalize_embeddings=True)
    #print ("chunk_vectors")
    #print(chunk_vectors)
    index = faiss.IndexFlatIP(chunk_vectors.shape[1])
    index.add(chunk_vectors)
    return index

# generates questions from the chunks
def generate_questions(chunks):
    questions = []
    for chunk in chunks:
        questions.append(chunk)
    return questions

def generate_answers_without_context(question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can answer questions about the context provided."}
    ]
    messages.append({"role": "user", "content": f"Answer the question: {question}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content
   
   

def generate_answers_with_llm(question, contexts):
    
    # Generate answer with retrieved contexts
    context_text = "\n".join(contexts)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can answer questions about the context provided."},
    ]

    messages.append({"role": "user", "content": f"Answer the question: {question} based on the following context: {context_text}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content 

# saves chunks to vector database
def save_chunks_to_database(chunks, embedder, database_name):
    table = db.open_table(database_name)
    
    # Get expected embedding dimension from a sample
    sample_embedding = embedder.encode(["sample"], normalize_embeddings=True)[0]
    expected_dim = len(sample_embedding)
    #print(f"Expected embedding dimension for {database_name}: {expected_dim}")
    
    # Process chunks in batches for better performance
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        data_to_add = []
        
        for chunk in batch_chunks:
            # Clean the chunk text
            chunk_text = chunk["text"].strip()
            if not chunk_text:  # Skip empty chunks
                continue
                
            # Get the embedding and ensure it's a proper list
            embedding = embedder.encode([chunk_text], normalize_embeddings=True)[0]
            
            # Ensure embedding is a Python list of floats
            if hasattr(embedding, 'tolist'):
                vector_list = embedding.tolist()
            else:
                vector_list = list(embedding)
            
            # Verify it's a list of numbers with correct dimension
            if not isinstance(vector_list, list) or not all(isinstance(x, (int, float)) for x in vector_list):
                print(f"Warning: Invalid vector format for chunk: {chunk_text[:50]}...")
                continue
                
            # Check dimension consistency - this is critical for fixed-size lists
            if len(vector_list) != expected_dim:
                print(f"ERROR: Vector dimension mismatch. Expected {expected_dim}, got {len(vector_list)} for chunk: {chunk_text[:50]}...")
                print(f"Skipping this chunk to avoid schema violation")
                continue
                
            data_to_add.append({
                "vector": vector_list,
                "text": chunk_text,
                "page": 0,
                "chunk_id": chunk["chunk_id"]
            })
        
        if data_to_add:  # Only add if we have valid data
            try:
                table.add(data_to_add)
                #print(f"Added batch {i//batch_size + 1}: {len(data_to_add)} chunks to {database_name}")
            except Exception as e:
                print(f"Error adding batch to {database_name}: {e}")
                # Print first few vectors for debugging
                for j, item in enumerate(data_to_add[:3]):
                    print(f"Sample vector {j}: length={len(item['vector'])}, type={type(item['vector'][0]) if item['vector'] else 'empty'}")
                raise

# retrieves answers from the pdf
def retrieve_answers_from_database(database_name, questions, embedder, top_k=3):
    table = db.open_table(database_name)
    
    # Encode the first question and get 1D vector as list
    question_vector = embedder.encode([questions[0]], normalize_embeddings=True)[0]
    
    # Ensure it's a proper list
    if hasattr(question_vector, 'tolist'):
        question_vector = question_vector.tolist()
    else:
        question_vector = list(question_vector)
     
    results = table.search(question_vector, vector_column_name="vector").limit(top_k).to_pandas()
    return results['text'].tolist(), results['chunk_id'].tolist()

def debug_database(database_name):
    """Debug function to check database status"""
    try:
        # List all tables
        print("Available tables:", db.table_names())
        
        # Check if our table exists
        if database_name in db.table_names():
            table = db.open_table(database_name)
            print("Table schema:", table.schema)
            print("Table count:", table.count_rows())
            
            # Show a sample row
            if table.count_rows() > 0:
                sample = table.head(1).to_pandas()
                print("Sample row:")
                print(sample)
                
                # Check column names
                print("Column names:", list(sample.columns))
                
                # Check vector column specifically
                if 'vector' in sample.columns:
                    vector_sample = sample['vector'].iloc[0]
                    print(f"Vector column type: {type(vector_sample)}")
                    if hasattr(vector_sample, '__len__'):
                        print(f"Vector dimension: {len(vector_sample)}")
                else:
                    print("WARNING: No 'vector' column found!")
            else:
                print("Table exists but is empty")
        else:
            print("Table 'chess-knowledge-base' does not exist")
            
    except Exception as e:
        print(f"Database debug error: {e}")


def embed_document(document_path, database_name, embedder, chunk_size, overlap):
    doc = read_from_pdf(document_path)
    chunks = []
    for page in range(len(doc)):
        chunks_batch = chunk_pdf_page(doc,page, chunk_size, overlap)
        save_chunks_to_database(chunks_batch, embedder, database_name)
        chunks.extend(chunks_batch)
        # need the print to stay on the same line and clear the previous line and simulate a progress bar
        print("\r", end="")
        print(f"Embedded page {page} of {len(doc)}...", end="\r")
    print(f"Document embedded successfully with parameters: chunk_size={chunk_size}, overlap={overlap} for database {database_name}")
    return chunks

# compute chunking metrics: recall, precision, mrr, time 
# Clear existing tables first to avoid schema conflicts
clear_existing_tables()
init_database_schema()

def generate_question_for_chunk(chunk):
    prompt = f"""
    Generate a question for the following chunk: {chunk["text"]} and return it as json
    the json should have the following fields:
    - question: the question
    - chunk_id: {chunk["chunk_id"]}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def run_metrics():
    for parameter in combo_parameters:

        print(f"Running with chunk size: {parameter['chunk_size']}, overlap: {parameter['overlap']} and embedder: {parameter['embedder_name']}...")
        # chunk the document
        chunks = embed_document("data/annual-report.pdf", parameter["table_name"], parameter["embedder"], parameter["chunk_size"], parameter["overlap"])
       
        # generate questions for the chunks, ONLY GENERATE 10 QUESTIONS
        questions = []
        for chunk in chunks[:20]:
            question = generate_question_for_chunk(chunk)
            question = question.replace("```json", "").replace("```", "")
            json_question = json.loads(question)
            #print(f"Question: {question}")
            questions.append({
                "question": json_question["question"],
                "chunk_id": json_question["chunk_id"]
            })

        # generate answers for the questions and compute metrics
        recall_1 = 0
        recall_3 = 0
        recall_5 = 0
        recall_10 = 0
        mrr_total = 0

        for question in questions:
            contexts, chunk_ids = retrieve_answers_from_database(parameter["table_name"], [question["question"]], parameter["embedder"], 10)

            #print(f"Chunk IDs: {chunk_ids}")
            #print(f"Question: { question }")
            #print(f"Answer: {contexts}")

            if chunk_ids[0] == question["chunk_id"]:
                recall_1 += 1

            i = 0
            for chunk_id in chunk_ids:
                if chunk_id == question["chunk_id"]:
                    if i < 3:
                        recall_3 += 1
                    if i < 5:
                        recall_5 += 1
                    if i < 10:
                        recall_10 += 1
                    i += 1
                    mrr_total += 1 / (chunk_ids.index(question["chunk_id"]) + 1)
   
        #time = end_time - start_time
        print(f"Chunk size: {parameter['chunk_size']}, Overlap: {parameter['overlap']}, Context Recall: {recall_1/len(questions)}, Context Recall 3: {recall_3/len(questions)}, Context Recall 5: {recall_5/len(questions)}, Context Recall 10: {recall_10/len(questions)}, MRR: {mrr_total / len(questions)}")



if __name__ == "__main__":
    run_metrics()