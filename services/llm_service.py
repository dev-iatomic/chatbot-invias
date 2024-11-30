from fastapi import FastAPI, APIRouter, Request
import os
import pickle
from langchain.vectorstores import FAISS  # Importación de FAISS desde langchain.vectorstores
from google.cloud import storage
from PyPDF2 import PdfReader
from langchain.schema import Document
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
import faiss  # Para manejo de particiones y optimización de vectores
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

router = APIRouter()

# Paso 3: Autenticación con Google Cloud Platform
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Configuración
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PROJECT_ID = os.getenv("PROJECT_ID", "bingo-433202")
LOCATION = os.getenv("LOCATION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET", "rag-docs-llama-436921")
LOCAL_PDF_FOLDER = os.getenv("LOCAL_PDF_FOLDER")
CACHE_FOLDER = os.getenv("CACHE_FOLDER")
EMBEDDINGS_CACHE_FOLDER = "/mnt/d/IAtomic/invias/chatbot/chatbot-invias/content/cache_modelo/embeddings"#os.getenv("EMBEDDINGS_CACHE_FOLDER")
EMBEDDINGS_CACHE_PATH = f'{EMBEDDINGS_CACHE_FOLDER}/embeddings.pkl'

# Autenticarse utilizando las credenciales del archivo JSON
CREDENTIALS_PATH = os.getenv("CREDENCIALES_PATH", "rag-docs-llama-436921")
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language"
])

# Refrescar las credenciales para asegurarse de tener el token correcto
credentials.refresh(Request())


def download_pdfs():
    # Inicializar el cliente de Google Cloud Storage con las credenciales y el proyecto
    storage_client = storage.Client(credentials=credentials, project=PROJECT_ID)
    
    # Crear las carpetas locales si no existen
    if not os.path.exists(LOCAL_PDF_FOLDER):
        os.makedirs(LOCAL_PDF_FOLDER)

    # Inicializar el cliente de Google Cloud Storage
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Paso 5: Obtener los nombres de los archivos PDF en el bucket
    blobs = list(bucket.list_blobs())  # Convertir el generador en una lista
    pdf_files_in_bucket = {os.path.basename(blob.name) for blob in blobs if blob.name.endswith('.pdf')}
    
    # Obtener los archivos existentes en la carpeta local
    existing_files_in_local = {f for f in os.listdir(LOCAL_PDF_FOLDER) if f.endswith('.pdf')}
    
    # Verificar si todos los archivos PDF ya están presentes localmente
    if pdf_files_in_bucket == existing_files_in_local:
        print('Todos los archivos PDF ya están descargados. No se descargará nada.')
    else:
        # Descargar los archivos que no existen localmente
        for blob in blobs:
            if blob.name.endswith('.pdf'):
                local_path = os.path.join(LOCAL_PDF_FOLDER, os.path.basename(blob.name))
                if os.path.basename(blob.name) not in existing_files_in_local:
                    blob.download_to_filename(local_path)
                    print(f'Descargado {blob.name} a {local_path}')
                else:
                    print(f'El archivo {blob.name} ya existe en {local_path}. No se descargará.')


def extract_text_pdfs():
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)

    # Paso 6: Lectura de los PDFs y extracción de texto utilizando PyPDF2
    documents = []
    for pdf_file in os.listdir(LOCAL_PDF_FOLDER):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(LOCAL_PDF_FOLDER, pdf_file)
            cache_path = os.path.join(CACHE_FOLDER, f'{pdf_file}.pkl')

            # Verificar si el archivo ya está en la cache
            if os.path.exists(cache_path):
                print(f'Cargando documentos procesados desde cache para {pdf_file}')
                with open(cache_path, 'rb') as cache_file:
                    cached_documents = pickle.load(cache_file)
                    documents.extend(cached_documents)
            else:
                print(f'Procesando y guardando en cache {pdf_file}')
                pdfreader = PdfReader(pdf_path)
                processed_documents = []
                try:
                    for i, page in enumerate(pdfreader.pages):
                        content = page.extract_text()
                        if content:
                            document = Document(
                                page_content=content,
                                metadata={"file_name": pdf_file, "page": i + 1}
                            )
                            processed_documents.append(document)
                            documents.append(document)
                except Exception as e:
                    if 'PyCryptodome is required' in str(e):
                        print(f"Advertencia: No se pudo procesar el archivo {pdf_file} debido a la falta de PyCryptodome para AES. Se omitira este archivo.")
                        continue

                # Guardar los documentos procesados en cache
                with open(cache_path, 'wb') as cache_file:
                    pickle.dump(processed_documents, cache_file)
    return documents

def extract_documents_embeddings(documents):

    if not os.path.exists(EMBEDDINGS_CACHE_FOLDER):
        os.makedirs(EMBEDDINGS_CACHE_FOLDER)

    # Paso 7: División del texto en fragmentos para el proceso de recuperación
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_tokens = 1000
    splitter = CharacterTextSplitter(chunk_size=max_tokens // 2, chunk_overlap=200, separator='\n')
    chunks = [splitter.split_text(doc.page_content) for doc in documents if len(doc.page_content) > 0]
    chunks = [chunk for sublist in chunks for chunk in sublist]  # Aplanar la lista de listas

    # Paso 8: Configuración de Typesense para el almacenamiento de vectores y búsqueda
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document", credentials=credentials)

    # Verificar si los embeddings ya están en la cache
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        print('Cargando embeddings desde cache')
        with open(EMBEDDINGS_CACHE_PATH, 'rb') as cache_file:
            doc_embeddings = pickle.load(cache_file)
    else:
        print('Generando embeddings y guardando en cache')
        # Crear los embeddings para los fragmentos en lotes para evitar exceder la cuota
        doc_embeddings = []
        batch_size = 20  # Dividir en lotes más pequeños para evitar exceder la cuota

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_embeddings = embeddings.embed_documents(batch)
                doc_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error al procesar el lote {i // batch_size + 1}: {e}. Intentando nuevamente...")
                # Reintento con backoff exponencial en caso de error
                import time
                for retry in range(3):
                    time.sleep(2 ** retry)
                    try:
                        batch_embeddings = embeddings.embed_documents(batch)
                        doc_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        print(f"Reintento {retry + 1} fallido para el lote {i // batch_size + 1}: {e}")
                        if retry == 2:
                            print(f"Error persistente en el lote {i // batch_size + 1}, omitiendo este lote.")

        # Guardar los embeddings en cache
        with open(EMBEDDINGS_CACHE_PATH, 'wb') as cache_file:
            pickle.dump(doc_embeddings, cache_file)

    return np.array(doc_embeddings), embeddings, chunks

def response_model(doc_embeddings, embeddings, chunks):
    # Crear el índice FAISS
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    # Almacenar el índice en FAISS
    docsearch = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({str(i): Document(page_content=chunks[i]) for i in range(len(chunks))}),
        index_to_docstore_id={i: str(i) for i in range(len(chunks))}
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )

    # Paso 12: El template proporcionado
    template = """
    Eres María, una asistente amigable y entusiasta especializada en el Programa Caminos Comunitarios para la Paz.
    Debes seguir estas pautas en tus respuestas:

    1. PERSONALIDAD:
    - Usa un tono casual y cercano
    - NO repitas saludos si ya estás en una conversación
    - Usa emojis ocasionalmente para dar calidez
    - Muestra empatía y entusiasmo por el programa

    2. ESTRUCTURA DE RESPUESTA:
    - Responde directamente a la pregunta sin saludos innecesarios
    - Da la información de manera conversacional
    - Si es una lista o proceso, divídelo en pasos
    - Concluye con una invitación a seguir preguntando

    3. LÍMITES:
    - Si no tienes la información en el contexto, di: "¡Ups! Esa información no la tengo en mi base de datos. Te sugiero comunicarte con la línea de atención de Invías al 601 377 0600 para más detalles."
    - Para temas técnicos específicos, recomienda consultar directamente con Invías

    4. CONTEXTO DISPONIBLE:
    {context}

    5. PREGUNTA DEL USUARIO:
    {question}

    Responde directamente a la pregunta sin repetir saludos o presentaciones. Mantén un tono conversacional pero ve al punto.
    """
    prompt = PromptTemplate.from_template(template)


    # Crear la cadena QA usando RunnableParallel
    chain = (
        RunnableParallel(
            context=RunnableLambda(lambda docs: "\n\n".join([
                f"[Página {doc.metadata.get('page', 'N/A')} de {doc.metadata.get('file_name', 'N/A')}]: {doc.page_content}"
                for doc in docs if isinstance(doc, Document)
            ])),
            question=RunnablePassthrough()
        )
        | prompt
        | ChatVertexAI(
            temperature=0.3,
            model_name="gemini-1.5-flash",
            max_output_tokens=2048,
        )
        | StrOutputParser()
    )


    return retriever, chain

# Incializando
download_pdfs()
documents = extract_text_pdfs()
doc_documents, embeddings, chunks = extract_documents_embeddings(documents)
retriever, chain = response_model(doc_documents, embeddings, chunks)


async def process_message(message: str) -> str:
    # Recuperar documentos relevantes y ejecutar la cadena de QA
    docs = retriever.get_relevant_documents(message)
    response = chain.invoke({"context": docs, "question": message})
    return response
