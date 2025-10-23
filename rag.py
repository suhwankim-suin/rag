# rag.py
import os
# from dotenv import load_dotenv

# load_dotenv()

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

import retriever

# 모델 초기화
llm = ChatOpenAI( model = 'gpt-4o-mini' )

# 사용자 메시지 처리 함수
def get_ai_response( messages, docs ):
    response = retriever.document_chain.stream( {
        "messages": messages,
        "context": docs,
    } )

    for chunk in response:
        yield chunk 

# Streamlit 앱
st.title( 'GPT-4o LangChain Chat' )

# Streamlit session_state에 메시지 저장
if 'messages' not in st.session_state:
    st.session_state[ 'messages' ] = [
        SystemMessage( '너는 문서를 기반해 답변하는 도시 정책 전문가야.' ),
    ]

# Streamlit 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance( msg, SystemMessage ):
            st.chat_message( "system" ).write( msg.content )
        elif isinstance( msg, AIMessage ):
            st.chat_message( "assistant" ).write( msg.content )
        elif isinstance( msg, HumanMessage ):
            st.chat_message( "user" ).write( msg.content )

# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message( "user" ).write( prompt )                       # 사용자 메시지 출력
    st.session_state.messages.append( HumanMessage( prompt ) )      # 사용자 메시지 저장

    print( f"user\t: {prompt}" )
    augmented_query = retriever.query_qugmentation_chain.invoke( {
        "messages": st.session_state[ "messages" ],
        "query": prompt,
    } )
    print( f'augmented_query\t {augmented_query}' )

    # 관련 문서 검색
    print( "관련 문서 검색" )
    docs = retriever.retriever.invoke( f'{prompt}\n{augmented_query}' )

    for doc in docs:
        print( '-'*100 )
        print( doc )

        with st.expander( f"**문서:** {doc.metadata.get( 'source', '알 수 없음')}" ):
            # 파일명과 페이지 정보 표시
            st.write( f"**page:**{doc.metadata.get( 'page', '' )}" )
            st.write( doc.page_content )
    print( '='*110 )

    with st.spinner( f"AI가 답변을 준비중입니다... '{augmented_query}'" ):
        response = get_ai_response( st.session_state[ "messages" ], docs )
        result = st.chat_message( "assistant" ).write_stream( response )
    
    st.session_state[ "messages" ].append( AIMessage( result ) )
 

 
