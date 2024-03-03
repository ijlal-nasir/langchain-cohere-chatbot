from app_chain import get_chatbot_chain
import streamlit as st

 
chain = get_chatbot_chain()
 
# URL for the logo of the assistant bot
# We need it as a separate variable because it's used in multiple places
bot_logo = 'https://pbs.twimg.com/profile_images/1739538983112048640/4NzIg1h6_400x400.jpg'
 
# We use st.session_state and fill in the st.session_state.messages list
# It's empty in the beginning, so we add the first message from the bot
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "bot",
                                     "content": "Hello, how can I help?"}]
 
# Then we show all the chat messages in Markdown format
for message in st.session_state['messages']:
    if message["role"] == 'bot':
        with st.chat_message(message["role"], avatar=bot_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
 
# We ask for the user's question, append it to the messages and show below
if query := st.chat_input("Please ask your question here:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
 
    # We create a new chat message and launch the "chain" for the answer
    with st.chat_message("assistant", avatar=bot_logo):
        message_placeholder = st.empty()
        result = chain.invoke({"question": query})
        message_placeholder.markdown(result['answer'])
 
    # We also add the answer to the messages history
    st.session_state.messages.append({"role": "bot", "content": result['answer']})