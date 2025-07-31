import os
import torch
import gradio as gr
import speech_recognition as sr
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
import time

# --------------------- Configurations ---------------------
PDF_PATH = "english_dataset_cleaned.pdf"
REFERENCE_SPEAKER = "resources/trump.mp3"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_API_KEY = "AIzaSyCHkRovviGK45wTHjIrHDtifU-dUhITeh0"
BASE_SPEAKER = "EN-US"
TOP_K = 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs_v2"
SRC_PATH = f"{OUTPUT_DIR}/tmp.wav"
SPEED = 0.85
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------- Voice Model Setup ---------------------
ckpt_converter = 'checkpoints_v2/converter'
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=DEVICE)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
target_se, _ = se_extractor.get_se(REFERENCE_SPEAKER, tone_color_converter, vad=True)

# --------------------- RAG Utilities ---------------------
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
    subquestions = [query]
    output_parser = StrOutputParser()
    all_answers = []

    for subq in subquestions:
        docs = retriever.get_relevant_documents(subq)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = PromptTemplate.from_template(
            "You are a helpful and elaborate assistant. Use the context below to answer in detailed and informative paragraphs:\n{context}\n\nQuestion: {question}"
        ).format_prompt(context=context, question=subq)

        response = llm.invoke(prompt)
        answer = output_parser.invoke(response)
        all_answers.append(answer)

    summary_prompt = PromptTemplate.from_template(
        "You are a smart assistant. Given the answers below, produce a final detailed answer:\n\n{answers}"
    ).format_prompt(answers="\n\n".join(all_answers))

    final_response = llm.invoke(summary_prompt)
    return output_parser.invoke(final_response)

# --------------------- TTS Function ---------------------
def speak(text):
    model = TTS(language="EN", device=DEVICE)
    speaker_ids = dict(model.hps.data.spk2id)
    matched_key = next((k for k in speaker_ids if k.lower() == BASE_SPEAKER.lower()), None)
    if not matched_key:
        raise ValueError(f"Speaker '{BASE_SPEAKER}' not found.")
    speaker_id = speaker_ids[matched_key]

    model.tts_to_file(text, speaker_id, SRC_PATH, speed=SPEED)
    output_path = f"{OUTPUT_DIR}/final_output.wav"
    tone_color_converter.convert(
        audio_src_path=SRC_PATH,
        src_se=torch.load(f'checkpoints_v2/base_speakers/ses/{matched_key.lower().replace("_", "-")}.pth', map_location=DEVICE),
        tgt_se=target_se,
        output_path=output_path,
        message="@MyShell"
    )
    return output_path

# --------------------- Initialization ---------------------
print("Initializing models...")
docs = load_pdf_text(PDF_PATH)
chunks = chunk_documents(docs)
vectorstore = create_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = setup_llm()

# --------------------- Gradio Interface ---------------------
class MicState:
    is_listening = False

def toggle_listen():
    MicState.is_listening = not MicState.is_listening
    return "Listening..." if MicState.is_listening else "Click to talk"

def listen_and_answer():
    if not MicState.is_listening:
        return "", "", None, "Click to talk"

    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        time.sleep(1.5)
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
            question = r.recognize_google(audio)
        except:
            return "", "I didn’t catch that. Try again.", None, "Click to talk"

    answer = agentic_rag(question, retriever, llm)
    answer_audio = speak(answer)
    return question, answer, answer_audio, "Click to talk"

with gr.Blocks() as app:
    gr.Markdown("""
    <h1 style='text-align: center;'>🇺🇸 Trump Voice AI — Ask Me Anything!</h1>
    """)

    user_text = gr.Textbox(label="🗣️ You Said", interactive=False)
    out_text = gr.Textbox(label="🧠 Trump's Answer", interactive=False)
    out_audio = gr.Audio(label="🎧 Trump Speaks", autoplay=True)
    status = gr.Textbox(label="🔄 Status", value="Click to talk", interactive=False)
    btn = gr.Button("🎙️ Talk to Trump")

    # Connect Buttons
    btn.click(fn=toggle_listen, inputs=[], outputs=[status])
    btn.click(fn=listen_and_answer, inputs=[], outputs=[user_text, out_text, out_audio, status])

    # Greet on App Load
    welcome_audio_path = speak("Hello guys! It's me Trump. How should I motivate you today?")
    gr.Audio(value=welcome_audio_path, autoplay=True, label="Trump's Welcome")

app.queue().launch(share=True)
