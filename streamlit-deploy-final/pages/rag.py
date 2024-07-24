import streamlit as st
from langchain_core.messages import ChatMessage
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
import os


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.set_page_config(page_title="RAG 시스템 💬", page_icon="💬")
st.title("RAG 시스템 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

if "vectorestore" not in st.session_state:
    st.session_state["vectorstore"] = None


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

    return vectorstore


with st.sidebar:
    clear_btn = st.button("대화내용 초기화")

    # 파일 업로드
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
    )

    retrieve_count = st.number_input("검색 결과 수", min_value=1, max_value=10, value=5)
    retrieve_algorithm = st.selectbox("검색 방법", ["기본", "mmr"])

    apply_btn = st.button("설정 적용")


if file:
    # 벡터스토어 생성
    st.session_state["vectorstore"] = embed_file(file)


def create_retriever(retrieve_count, algorithm="기본"):
    if st.session_state["vectorstore"]:
        if algorithm == "기본":
            return st.session_state["vectorstore"].as_retriever(
                search_kwargs={"k": retrieve_count}
            )
        else:
            return st.session_state["vectorstore"].as_retriever(
                search_type=algorithm, search_kwargs={"k": retrieve_count}
            )
    else:
        return None


if clear_btn:
    retriever = st.session_state["messages"].clear()


print_history()

msg_container = st.empty()

if user_input := st.chat_input():
    add_history("user", user_input)
    if st.session_state["vectorstore"] is None:
        msg_container.error("파일을 업로드해주세요.")
    else:
        st.chat_message("user").write(user_input)
        retriever = create_retriever(retrieve_count, retrieve_algorithm)

        with st.chat_message("assistant"):

            if retriever:
                # 검색된 결과 -> List[Document]
                ai_answer = retriever.invoke(user_input)

                for i, doc in enumerate(ai_answer):
                    expander = st.expander(
                        f"{i+1}. {doc.metadata['source'].split('/')[-1]}, page: {doc.metadata['page']}"
                    )
                    expander.markdown(doc.page_content)
