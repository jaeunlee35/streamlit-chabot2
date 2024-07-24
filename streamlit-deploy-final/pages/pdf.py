import glob
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
# from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import load_prompt
import os



# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.set_page_config(page_title="PDF 기반 챗봇 💬", page_icon="💬")
st.title("PDF 기반 챗봇 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # RAG 체인 초기화
    st.session_state["chain"] = None

if "retriever" not in st.session_state:
    st.session_state["retriever"] = None


def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # 1단계: 문서 로드
    loader = PyMuPDFLoader(file_path)
    # loader = UnstructuredFileLoader(file_path)

    # 2단계: 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    split_docs = loader.load_and_split(text_splitter=text_splitter)

    # 3단계: 임베딩
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # 4단계: 벡터 저장소에 저장
    vectorstore = FAISS.from_documents(split_docs, embedding=cached_embeddings)

    # 5단계: 검색기 생성
    retriever = vectorstore.as_retriever()
    return retriever


# 체인을 생성합니다.
def create_chain(retriever, model_name="gpt-4o", prompt=None):
    # 체인을 생성합니다.
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | ChatOpenAI(model_name=model_name)
        | StrOutputParser()
    )
    return chain


with st.sidebar:
    clear_btn = st.button("대화내용 초기화")
    # LLM 선택
    model_selection = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo"]
    )

    # RAG 프롬프트 선택
    prompt_files = glob.glob("pages/prompts/*.yaml")
    prompt_selection = st.selectbox(
        "RAG 프롬프트 선택",
        prompt_files,
    )

    # 파일 업로드
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
    )

    apply_btn = st.button("설정 적용")


if file:
    # 검색기를 생성
    st.session_state["retriever"] = embed_file(file)
    # 체인을 생성
    # RAG 전용 프롬프트
    st.session_state["chain"] = create_chain(
        st.session_state["retriever"],
        model_name=model_selection,
        prompt=load_prompt(prompt_selection, encoding="utf-8"),
    )

if apply_btn:
    if st.session_state["retriever"] is not None:
        st.session_state["chain"] = create_chain(
            st.session_state["retriever"],
            model_name=model_selection,
            prompt=load_prompt(prompt_selection, encoding="utf-8"),
        )

if clear_btn:
    retriever = st.session_state["messages"].clear()


print_history()

msg_container = st.empty()

if user_input := st.chat_input():
    chain = st.session_state["chain"]
    if chain is not None:
        add_history("user", user_input)
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            chat_container = st.empty()

            stream_response = chain.stream(user_input)  # 문서에 대한 질의
            ai_answer = ""
            for chunk in stream_response:
                ai_answer += chunk
                chat_container.markdown(ai_answer)
            add_history("ai", ai_answer)
    else:
        msg_container.error("파일을 업로드해주세요.")
