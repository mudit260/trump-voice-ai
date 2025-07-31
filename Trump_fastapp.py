import os
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --------------------- Config ---------------------
PDF_PATH = "english_dataset_cleaned.pdf"
REFERENCE_SPEAKER = "resources/trump.mp3"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_API_KEY = "AIzaSyCHkRovviGK45wTHjIrHDtifU-dUhITeh0"  # Replace with your key
BASE_SPEAKER = "EN-US"
TOP_K = 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs_v2"
SRC_PATH = f"{OUTPUT_DIR}/tmp.wav"
SPEED = 0.85
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------- Load Voice + SE ---------------------
ckpt_converter = 'checkpoints_v2/converter'
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=DEVICE)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
target_se, _ = se_extractor.get_se(REFERENCE_SPEAKER, tone_color_converter, vad=True)

# --------------------- RAG Setup ---------------------
def load_pdf_text(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()

def chunk_documents(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_vectorstore(chunks):
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return FAISS.from_documents(chunks, embed_model)

def setup_llm():
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        convert_system_message_to_human=True,
        temperature=0.6
    )

def agentic_rag(query, retriever, llm):
    output_parser = StrOutputParser()
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use the context to answer the question:\n{context}\n\nQuestion: {question}"
    ).format_prompt(context=context, question=query)

    response = llm.invoke(prompt)
    return output_parser.invoke(response)

# --------------------- TTS Function ---------------------
def speak(text):
    model = TTS(language="EN", device=DEVICE)
    speaker_ids = dict(model.hps.data.spk2id)
    matched_key = next((k for k in speaker_ids if k.lower() == BASE_SPEAKER.lower()), None)
    speaker_id = speaker_ids[matched_key]

    model.tts_to_file(text, speaker_id, SRC_PATH, speed=SPEED)
    output_path = f"{OUTPUT_DIR}/final_output.wav"
    tone_color_converter.convert(
        audio_src_path=SRC_PATH,
        src_se=torch.load(f'checkpoints_v2/base_speakers/ses/{matched_key.lower().replace("_", "-")}.pth', map_location=DEVICE),
        tgt_se=target_se,
        output_path=output_path,
        message="@TrumpVoice"
    )
    return output_path

# --------------------- Init Models Once ---------------------
print("Initializing models...")
docs = load_pdf_text(PDF_PATH)
chunks = chunk_documents(docs)
vectorstore = create_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = setup_llm()

# --------------------- FastAPI ---------------------
app = FastAPI()

class QueryInput(BaseModel):
    text: str

@app.post("/ask")
async def ask_trump(query: QueryInput):
    answer = agentic_rag(query.text, retriever, llm)
    audio_path = speak(answer)
    return {
        "question": query.text,
        "answer": answer,
        "audio_url": "/audio"
    }

@app.get("/audio")
def get_audio():
    return FileResponse("outputs_v2/final_output.wav", media_type="audio/wav")

@app.get("/")
def root():
    return {"message": "🚀 Trump Voice API is ready!"}
