"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata

import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct

############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        print("Retrieverは既に作成済みです")
        logger.info("Retrieverは既に作成済みです")
        return

    print("データソースの読み込み開始")
    logger.info("データソースの読み込み開始")
    docs_all = load_data_sources()
    print(f"データソースの読み込み完了: {len(docs_all)}件")
    logger.info(f"データソースの読み込み完了: {len(docs_all)}件")

    print("文字列調整処理開始")
    logger.info("文字列調整処理開始")
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    print("文字列調整処理完了")
    logger.info("文字列調整処理完了")

    print("埋め込みモデルの用意開始")
    logger.info("埋め込みモデルの用意開始")
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    print("埋め込みモデルの用意完了")
    logger.info("埋め込みモデルの用意完了")

    print("チャンク分割オブジェクト作成開始")
    logger.info("チャンク分割オブジェクト作成開始")
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )
    print("チャンク分割オブジェクト作成完了")
    logger.info("チャンク分割オブジェクト作成完了")

    print("チャンク分割処理開始")
    logger.info("チャンク分割処理開始")
    splitted_docs = text_splitter.split_documents(docs_all)
    print(f"チャンク分割処理完了: {len(splitted_docs)}件")
    logger.info(f"チャンク分割処理完了: {len(splitted_docs)}件")

    print("ベクターストア作成開始")
    logger.info("ベクターストア作成開始")
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)
    print("ベクターストア作成完了")
    logger.info("ベクターストア作成完了")

    print("Retriever作成開始")
    logger.info("Retriever作成開始")
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": 3})
    print("Retriever作成完了")
    logger.info("Retriever作成完了")


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)
    
    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    logger = logging.getLogger(ct.LOGGER_NAME)
    if os.path.isdir(path):
        print(f"ディレクトリ探索: {path}")
        logger.info(f"ディレクトリ探索: {path}")
        files = os.listdir(path)
        for file in files:
            full_path = os.path.join(path, file)
            recursive_file_check(full_path, docs_all)
    else:
        print(f"ファイル検出: {path}")
        logger.info(f"ファイル検出: {path}")
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        print(f"対応拡張子: {file_extension} 読み込み開始: {path}")
        logger.info(f"対応拡張子: {file_extension} 読み込み開始: {path}")
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        print(f"読み込み完了: {path} ドキュメント数: {len(docs)}")
        logger.info(f"読み込み完了: {path} ドキュメント数: {len(docs)}")
        docs_all.extend(docs)
    else:
        print(f"未対応拡張子: {file_extension} スキップ: {path}")
        logger.info(f"未対応拡張子: {file_extension} スキップ: {path}")

def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s