import os
import base64
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from vector_db import query_documents, get_collection_stats
from openai import OpenAI

# Store conversation history in memory
conversation_history = []


def get_llm():
    """Initialize LangChain LLM with OpenRouter"""
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7
    )
    return llm


def create_rag_chain():
    """Create a RAG (Retrieval Augmented Generation) chain with conversation memory"""
    llm = get_llm()

    # Format conversation history as a string
    history_text = ""
    if conversation_history:
        history_text = "Previous conversation:\n"
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"
        history_text += "\n"

    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""{history}You are a helpful langchain assistant, called the group 31 tech talk agent. Use the following context to answer the user's question. Please answer in a concise paragraph format, do not use bullets, markdown or em dashes. Your audience is a third year business class who is learning about langchain for the first time in class. Remember the previous conversation when relevant.

Context:
{context}

User Question:
{question}

Answer:"""
    )

    chain = prompt_template | llm | StrOutputParser()
    return chain


def get_context_from_knowledge_base(query: str, max_docs: int = 3) -> str:
    """Retrieve relevant context from the knowledge base"""
    results = query_documents(query, n_results=max_docs)

    if not results or not results.get("documents") or len(results["documents"]) == 0:
        return "No relevant documents found in knowledge base."

    documents = results["documents"][0]
    context = "\n\n".join([f"Document: {doc}" for doc in documents])
    return context


def chat_with_agent(user_message: str) -> str:
    """Process user message and generate response using RAG with conversation memory"""
    global conversation_history

    try:
        # Get relevant context from knowledge base
        context = get_context_from_knowledge_base(user_message)

        # Format conversation history
        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for msg in conversation_history:
                if isinstance(msg, HumanMessage):
                    history_text += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"Assistant: {msg.content}\n"
            history_text += "\n"

        # Create and run the RAG chain
        chain = create_rag_chain()
        response = chain.invoke({
            "history": history_text,
            "context": context,
            "question": user_message
        })

        # Store messages in conversation history
        conversation_history.append(HumanMessage(content=user_message))
        conversation_history.append(AIMessage(content=response))

        # Keep history manageable (last 20 messages = 10 turn conversation)
        if len(conversation_history) > 20:
            conversation_history.pop(0)
            conversation_history.pop(0)

        return response
    except Exception as e:
        return f"Error processing message: {str(e)}"


def get_audio_response(text_response: str) -> bytes:
    """Convert text response to speech using gpt-audio-mini by having it read the text"""
    try:
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

        # Request audio output from the model with streaming enabled
        # Key: Tell it to READ the text, not respond to it
        response = client.chat.completions.create(
            model="openai/gpt-audio-mini",
            modalities=["text", "audio"],
            audio={
                "voice": "shimmer",  # options: alloy, echo, fable, onyx, nova, shimmer
                "format": "pcm16"  # Required for streaming
            },
            messages=[
                {
                    "role": "system",
                    "content": "You are a text-to-speech assistant. Read the following text aloud exactly as written, with no changes, additions, or commentary."
                },
                {
                    "role": "user",
                    "content": f"Read this text aloud: {text_response}"
                }
            ],
            max_tokens=2000,
            stream=True  # Required for audio output
        )

        # Handle streaming response
        audio_bytes = b""
        for chunk in response:
            try:
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    # Handle both object and dict access patterns
                    delta = None
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    elif isinstance(choice, dict) and 'delta' in choice:
                        delta = choice['delta']

                    if delta:
                        audio = None
                        if hasattr(delta, 'audio'):
                            audio = delta.audio
                        elif isinstance(delta, dict) and 'audio' in delta:
                            audio = delta['audio']

                        if audio:
                            audio_data = None
                            if hasattr(audio, 'data'):
                                audio_data = audio.data
                            elif isinstance(audio, dict) and 'data' in audio:
                                audio_data = audio['data']

                            if audio_data:
                                # Audio data comes base64 encoded, decode it
                                audio_chunk = base64.b64decode(audio_data)
                                audio_bytes += audio_chunk
            except (AttributeError, KeyError, TypeError) as chunk_error:
                # Skip chunks that don't have audio data
                continue

        return audio_bytes if audio_bytes else None

    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def clear_conversation():
    """Clear conversation history for a new chat"""
    global conversation_history
    conversation_history = []
    return {"success": True, "message": "Chat cleared"}


def process_uploaded_text(text: str) -> dict:
    """Process and store uploaded text in knowledge base"""
    try:
        from vector_db import add_documents

        # Split text into chunks (simple approach)
        chunks = text.split("\n\n")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        if not chunks:
            return {"success": False, "message": "No content to upload"}

        # Add to database
        count = add_documents(chunks)

        return {
            "success": True,
            "message": f"Successfully uploaded {count} document chunks",
            "count": count
        }
    except Exception as e:
        return {"success": False, "message": f"Error uploading: {str(e)}"}


def get_kb_status() -> dict:
    """Get knowledge base statistics"""
    stats = get_collection_stats()
    return {
        "documents": stats.get("count", 0),
        "status": "Ready" if stats.get("count", 0) > 0 else "Empty"
    }