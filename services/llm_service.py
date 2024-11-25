# llm_service.py

import os
import unicodedata
from google.cloud import storage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

# Configuración
BUCKET_NAME = os.getenv("BUCKET_NAME")
TEMP_DIR = "./temp_pdfs"
TARGET_DOC_NAME = "doc_base_caja_herramientas.pdf"

os.makedirs(TEMP_DIR, exist_ok=True)

def get_pdfs_from_bucket(bucket_name):
    client = storage.Client()
    blobs = client.list_blobs(bucket_name)
    pdf_paths = []
    for blob in blobs:
        if blob.name.endswith(".pdf"):
            temp_path = os.path.join(TEMP_DIR, os.path.basename(blob.name))
            blob.download_to_filename(temp_path)
            pdf_paths.append(temp_path)
    return pdf_paths

def clean_text(text):
    # Normalizar el texto para eliminar caracteres problemáticos
    text = unicodedata.normalize('NFKD', text)
    # Reemplazar caracteres de control y caracteres inválidos
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' and c.isprintable())
    # Eliminar caracteres de sustitución (surrogates)
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return text

def split_text_into_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)
    # Limpiar el texto de cada chunk
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
    return chunks

def filter_unique_chunks(chunks):
    unique_chunks = []
    seen_texts = set()
    for chunk in chunks:
        chunk_text = chunk.page_content.strip()
        if chunk_text not in seen_texts:
            unique_chunks.append(chunk)
            seen_texts.add(chunk_text)
    return unique_chunks

def get_generative_ai_embeddings():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    return embeddings

def save_each_document_to_vectorstore(documents, embeddings):
    vectorstores = []
    retrievers = []
    for i, (doc_name, document) in enumerate(documents):
        vectorstore_dir = f"./vectorstore_{i}"
        if os.path.exists(vectorstore_dir):
            # Cargar el vectorstore existente
            vectorstoredb = Chroma(
                persist_directory=vectorstore_dir,
                embedding_function=embeddings
            )
        else:
            # Crear y guardar el vectorstore
            chunks = split_text_into_chunks(document)
            unique_chunks = filter_unique_chunks(chunks)
            vectorstoredb = Chroma.from_documents(
                documents=unique_chunks,
                embedding=embeddings,
                persist_directory=vectorstore_dir
            )
            vectorstoredb.persist()
        vectorstores.append(vectorstoredb)
        retriever = vectorstoredb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        retrievers.append(retriever)
    return retrievers

def load_vectorstores(embeddings, num_documents):
    vectorstores = []
    retrievers = []
    for i in range(num_documents):
        vectorstore_dir = f"./vectorstore_{i}"
        vectorstoredb = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=embeddings
        )
        vectorstores.append(vectorstoredb)
        retriever = vectorstoredb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        retrievers.append(retriever)
    return retrievers

def load_pdfs_to_list(pdf_paths):
    pdf_data = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        if os.path.basename(pdf) == TARGET_DOC_NAME:
            data = loader.load()
            filtered_data = data[7:]
            pdf_data.append((pdf, filtered_data))
        else:
            data = loader.load()
            pdf_data.append((pdf, data))
    return pdf_data

# Obtener los PDFs del bucket
pdf_files = get_pdfs_from_bucket(BUCKET_NAME)
pdf_data_list = load_pdfs_to_list(pdf_files)
embeddings = get_generative_ai_embeddings()

# Verificar si los vectorstores ya existen
vectorstore_dirs = [f"./vectorstore_{i}" for i in range(len(pdf_data_list))]
if all(os.path.exists(dir) for dir in vectorstore_dirs):
    # Cargar los vectorstores existentes
    retrievers = load_vectorstores(embeddings, len(pdf_data_list))
else:
    # Construir y guardar los vectorstores
    retrievers = save_each_document_to_vectorstore(pdf_data_list, embeddings)

# Configuración del LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

system_prompt = (
    "Eres María, una asistente amigable y entusiasta especializada en el Programa Caminos Comunitarios para la Paz.\n"
    "Debes seguir estas pautas en tus respuestas:\n\n"
    "1. PERSONALIDAD:\n"
    "- Usa un tono casual y cercano\n"
    "- NO repitas saludos si ya estás en una conversación\n"
    "- Usa emojis ocasionalmente para dar calidez\n"
    "- Muestra empatía y entusiasmo por el programa\n\n"
    "2. ESTRUCTURA DE RESPUESTA:\n"
    "- Responde directamente a la pregunta sin saludos innecesarios\n"
    "- Da la información de manera conversacional\n"
    "- Si es una lista o proceso, divídelo en pasos\n"
    "- Concluye con una invitación a seguir preguntando\n\n"
    "3. LÍMITES:\n"
    "- Si no tienes la información en el contexto, di: '¡Ups! Esa información no la tengo en mi base de datos. Te sugiero comunicarte con la línea de atención de Invías al 601 377 0600 para más detalles.'\n"
    "- Para temas técnicos específicos, recomienda consultar directamente con Invías\n\n"
    "4. CONTEXTO DISPONIBLE:\n"
    "{context}\n\n"
    "5. PREGUNTA DEL USUARIO:\n"
    "{question}\n\n"
    "Responde directamente a la pregunta sin repetir saludos o presentaciones. Mantén un tono conversacional pero ve al punto."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

chain = create_stuff_documents_chain(llm, prompt)

async def process_message(message: str) -> str:
    combined_context = []
    for retriever in retrievers:
        results = retriever.get_relevant_documents(message)
        combined_context.extend(results)
    consolidated_context = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in combined_context]
    response = chain.invoke({"context": consolidated_context, "question": message})
    return response
