import logging
from groq import Groq

logger = logging.getLogger(__name__)

MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a helpful document assistant. You answer questions based strictly on the provided PDF content.

Rules:
- Only use information from the provided context
- If the answer isn't in the context, say "I couldn't find that in the document"
- Be clear and concise
- If the question is a greeting or general, respond normally"""


def ask_groq(question, context, api_key, chat_history=None):
    """Send question + context to Groq and return the answer."""
    try:
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # include recent chat history for conversational context
        if chat_history:
            for msg in chat_history[-6:]:  # last 3 exchanges
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_msg = f"Relevant PDF content:\n{context}\n\nQuestion: {question}"
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )

        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            return "❌ Invalid API key. Please check your Groq API key in the sidebar."
        elif "429" in err:
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        else:
            logger.error(f"Groq error: {e}")
            return f"❌ Something went wrong: {str(e)}"
