"""
このファイルは、画面表示に特化した関数定義のファイルです。

"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct


############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """
    サイドバーに社内問い合わせ専用の説明のみ表示
    """
    with st.sidebar:
        st.markdown("---")


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    # メイン画面中央にAIの初期メッセージのみ表示
    with st.chat_message("assistant"):
        # 「st.success()」とすると緑枠で表示される
        st.success("こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。サイドバーで利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。")
        # 黄色背景の注意文を追加
        st.warning(f"{ct.WARNING_ICON} 具体的に入力したほうが期待通りの回答を得やすいです。")

    # サイドバーに社内問い合わせの機能説明のみ表示
    with st.sidebar:
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        st.code("【入力例】\n人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    # 会話ログのループ処理
    for message in st.session_state.messages:
        # 「message」辞書の中の「role」キーには「user」か「assistant」が入っている
        with st.chat_message(message["role"]):

            # ユーザー入力値の場合、そのままテキストを表示するだけ
            if message["role"] == "user":
                st.markdown(message["content"])
            
            # LLMからの回答の場合
            else:
                # 「社内問い合わせ」の場合の表示処理のみ残す
                st.markdown(message["content"]["answer"])

                # 参照元のありかを一覧表示
                if "file_info_list" in message["content"]:
                    # 区切り線の表示
                    st.divider()
                    # 「情報源」の文字を太字で表示
                    st.markdown(f"##### {message['content']['message']}")
                    # ドキュメントのありかを一覧表示
                    for file_info in message["content"]["file_info_list"]:
                        icon = utils.get_source_icon(file_info["source"])
                        if "page_number" in file_info:
                            st.info(f"{file_info['source']} (ページNo.{file_info['page_number'] + 1})", icon=icon)
                        else:
                            st.info(f"{file_info['source']}", icon=icon)





def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    # LLMからの回答を表示
    st.markdown(llm_response["answer"])

    # ユーザーの質問・要望に適切な回答を行うための情報が、社内文書のデータベースに存在しなかった場合
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        # 区切り線を表示
        st.divider()

        # 補足メッセージを表示
        message = "情報源"
        st.markdown(f"##### {message}")

        # 参照元のファイルパスの一覧を格納するためのリストを用意
        file_path_list = []
        file_info_list = []

        # LLMが回答生成の参照元として使ったドキュメントの一覧が「context」内のリストの中に入っているため、ループ処理
        for document in llm_response["context"]:
            file_path = document.metadata["source"]
            if file_path in file_path_list:
                continue
            if "page" in document.metadata:
                page_number = document.metadata["page"]
                file_info = {"source": file_path, "page_number": page_number}
            else:
                file_info = {"source": file_path}
            icon = utils.get_source_icon(file_path)
            if "page_number" in file_info:
                st.info(f"{file_info['source']} (ページNo.{file_info['page_number'] + 1})", icon=icon)
            else:
                st.info(f"{file_info['source']}", icon=icon)
            file_path_list.append(file_path)
            file_info_list.append(file_info)

    # 表示用の会話ログに格納するためのデータを用意
    # 社内問い合わせ専用
    # - 「answer」: LLMからの回答
    # - 「message」: 補足メッセージ
    # - 「file_path_list」: ファイルパスの一覧リスト
    content = {}
    content["answer"] = llm_response["answer"]
    # 参照元のドキュメントが取得できた場合のみ
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        content["message"] = message
        content["file_info_list"] = file_info_list
    return content