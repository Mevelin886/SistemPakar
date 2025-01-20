import os
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Konfigurasi keamanan
app.config['SECRET_KEY'] = os.urandom(24)

# Fungsi inisialisasi model dan dokumen
def initialize_chatbot():
    try:
        # Load dan proses PDF
        loader = PyPDFLoader("Data base rabies3.pdf")
        data = loader.load()
        
        # Pisah dokumen menjadi chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)

        # Buat vector store
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )

        # Buat retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )

        # Inisialisasi LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3, 
            max_tokens=None
        )

        # Template sistem
        system_template = (
            "Anda adalah asisten AI yang berfungsi sebagai sistem pakar untuk konsultasi Rabies. "
            "Gunakan konteks yang diberikan untuk menjawab pertanyaan dengan akurat, singkat, dan detail. "
            "Fokus pada memberikan informasi tentang pencegahan, gejala, diagnosis, dan penanganan Rabies. "
            "Berikan solusi praktis yang aman dan berbasis bukti untuk setiap permasalahan yang diajukan. "
            "Jika informasi tidak tersedia, sampaikan dengan jelas dan padat tanpa memberikan asumsi. "
            "Prioritaskan keselamatan manusia dan langkah-langkah pencegahan yang sesuai. "
            "berikan jawaban maksimal 4 kalimat"
            "\n\nKonteks: {context}"
        )

        # Buat prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Buat document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Buat retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever, 
            document_chain
        )

        return retrieval_chain

    except Exception as e:
        print(f"Kesalahan inisialisasi: {e}")
        return None

# Inisialisasi chatbot global
CHATBOT = initialize_chatbot()

# Memori percakapan
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        query = request.form.get('query', '').strip()
        
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query tidak boleh kosong'
            }), 400

        # Proses dengan retrieval chain
        result = CHATBOT.invoke({
            "input": query,
            "chat_history": conversation_memory.chat_memory.messages
        })

        # Simpan ke memori percakapan
        conversation_memory.save_context(
            {"input": query},
            {"output": result['answer']}
        )

        return jsonify({
            'status': 'success',
            'answer': result['answer'],
            'chat_history': [
                {
                    'role': msg.type,
                    'content': msg.content
                }
                for msg in conversation_memory.chat_memory.messages
            ]
        })

    except Exception as e:
        print(f"Error in chatbot: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Terjadi kesalahan dalam memproses pertanyaan'
        }), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True
    )
