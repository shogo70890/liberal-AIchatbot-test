import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
import os
import shutil
import glob

st.title("FITPLACE質問回答アプリ")

# アプリの概要や操作方法を表示
st.markdown("""
### 
このアプリでは、FITPLACEの資料を基に質問に答えます。
質問内容を入力して「送信」ボタンを押してください。
""")

# --- 会話履歴初期化 ---
chat_history = []

# 入力フォーム
input_text = st.text_input("質問内容を入力してください：", placeholder="例：ウェイトマシンのメンテナンスについて教えて")

# 送信ボタン
if st.button("送信"):
    if input_text:
        # 入力値をqueryとして利用
        query = input_text

        # RAGチェーンで回答を取得
        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })

        # 回答を表示
        st.write("回答:", result["answer"]["text"])

        # 根拠文書の表示
        if "source_documents" in result:
            st.write("根拠となった文書:")
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "不明")
                page = doc.metadata.get("page", "不明")
                st.write(f"・{source}（ページ: {page}）")

        # 会話履歴に追加
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=result["answer"]["text"])
        ])
    else:
        st.warning("質問内容を入力してください！")

# --- データ取得 ---
base_dir = "data"
docs = []

for file in os.listdir(base_dir):
    if not file.lower().endswith(".pdf"):
        continue

    file_path = os.path.join(base_dir, file)
    try:
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        # 各ページのmetadataにファイル名を追加
        for page in pages:
            page.metadata["source"] = file
        docs.extend(pages)
        print(f"{file} を処理しました")
    except Exception as e:
        print(f"{file} の読み込み失敗: {e}")

# docs には全PDFのページデータが格納されます

    except Exception as e:
        print(f" 通常部門（{theme}）: {file} の読み込み失敗: {e}")

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n",
)

splitted_pages = text_splitter.split_documents(docs)
print(f" 分割後のチャンク数: {len(splitted_pages)}")

embeddings = OpenAIEmbeddings()

# --- ベクターストア ---
all_db_path = os.path.join(base_dir, ".db")

all_db = None
if os.path.isdir(all_db_path):
    all_db = Chroma(persist_directory=all_db_path, embedding_function=embeddings)
    all_db.add_documents(splitted_pages)
    print(" .db に追加完了（追記）")
else:
    all_db = Chroma.from_documents(
        splitted_pages,
        embedding=embeddings,
        persist_directory=all_db_path
    )
    print(" .db を新規作成")

# --- ベクターストアの読み込み ---
base_dir = "data"
db_path = os.path.join(base_dir, ".db")

retriever = Chroma(persist_directory=db_path, embedding_function=embeddings).as_retriever()


# --- 会話履歴をもとに質問を再構成するためのプロンプト ---
question_generator_template = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"

question_generator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", question_generator_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- LLM（ストリーミング対応） ---
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.5,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


# --- コンテキストを元に回答を生成するプロンプト ---
question_answer_template = """
あなたは優秀な質問応答アシスタントです。以下のcontextを使用して質問に答えてください。
また答えが分からない場合は、無理に答えようとせず「分からない」という旨を答えてください。"

{context}
"""

question_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", question_answer_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- チェーンの構築 ---]
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=question_generator_prompt
)
from langchain.chains import LLMChain
question_answer_chain = LLMChain(
    llm=llm,
    prompt=question_answer_prompt,
    verbose=True
)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)