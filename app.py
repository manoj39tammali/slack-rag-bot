import os
import io
import requests as req
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import anthropic
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier

load_dotenv(dotenv_path="../.env")

app = Flask(__name__)

# Clients
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
verifier = SignatureVerifier(os.getenv("SLACK_SIGNING_SECRET"))

# In-memory RAG store (shared across all users)
pdf_chunks = []
pdf_name = ""


# ── RAG helpers (copied from rag-pdf-chat) ──────────────────────────

def extract_text_from_pdf(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def find_relevant_chunks(question, chunks, top_k=4):
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = chunks + [question]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    chunk_vectors = tfidf_matrix[:-1]
    question_vector = tfidf_matrix[-1]
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if similarities[i] > 0]


def ask_claude(question, chunks):
    relevant = find_relevant_chunks(question, chunks)
    if not relevant:
        return "I couldn't find relevant information in the loaded documents."
    context = "\n\n---\n\n".join(relevant)
    message = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""You are a helpful Slack assistant answering questions about uploaded documents.
Use only the context below. If the answer isn't there, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
            }
        ]
    )
    return message.content[0].text


# ── Slack event handler ──────────────────────────────────────────────

@app.route("/slack/events", methods=["POST"])
def slack_events():
    # Verify request is from Slack
    if not verifier.is_valid_request(request.get_data(), request.headers):
        return jsonify({"error": "invalid request"}), 403

    payload = request.json

    # Slack URL verification challenge
    if payload.get("type") == "url_verification":
        return jsonify({"challenge": payload["challenge"]})

    event = payload.get("event", {})
    event_type = event.get("type")

    # Bot mentioned: @bot <question>
    if event_type == "app_mention":
        channel = event["channel"]
        thread_ts = event.get("thread_ts", event.get("ts"))
        text = event.get("text", "")

        # Strip the bot mention from the text
        question = text.split(">", 1)[-1].strip()

        if not question:
            slack_client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="Please ask me a question! e.g. `@bot what is the refund policy?`"
            )
            return jsonify({"ok": True})

        if not pdf_chunks:
            slack_client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="No documents loaded yet. Use `/upload` and attach a PDF first."
            )
            return jsonify({"ok": True})

        # Answer using RAG
        answer = ask_claude(question, pdf_chunks)
        slack_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=answer
        )

    return jsonify({"ok": True})


# ── Slash command: /upload ───────────────────────────────────────────

@app.route("/slack/upload", methods=["POST"])
def slack_upload():
    global pdf_chunks, pdf_name

    # Verify request is from Slack
    if not verifier.is_valid_request(request.get_data(), request.headers):
        return jsonify({"error": "invalid request"}), 403

    user_id = request.form.get("user_id")
    channel_id = request.form.get("channel_id")

    # Prompt the user to upload a file
    slack_client.chat_postMessage(
        channel=channel_id,
        text=f"<@{user_id}> Please upload a PDF file in this channel and I'll process it for RAG. Make sure to share the file directly in the channel."
    )

    return jsonify({"response_type": "ephemeral", "text": "Please upload a PDF file in the channel."})


@app.route("/slack/file", methods=["POST"])
def slack_file_event():
    """Handle file uploads via event subscription."""
    if not verifier.is_valid_request(request.get_data(), request.headers):
        return jsonify({"error": "invalid request"}), 403

    payload = request.json

    if payload.get("type") == "url_verification":
        return jsonify({"challenge": payload["challenge"]})

    event = payload.get("event", {})

    if event.get("type") == "message" and event.get("files"):
        global pdf_chunks, pdf_name
        channel = event["channel"]

        for file_info in event["files"]:
            if file_info.get("filetype") != "pdf":
                slack_client.chat_postMessage(
                    channel=channel,
                    text="Only PDF files are supported."
                )
                continue

            # Download the file using bot token
            file_url = file_info["url_private"]
            headers = {"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"}
            response = req.get(file_url, headers=headers)

            if response.status_code != 200:
                slack_client.chat_postMessage(
                    channel=channel,
                    text="Failed to download the file. Please try again."
                )
                continue

            text = extract_text_from_pdf(response.content)
            if not text.strip():
                slack_client.chat_postMessage(
                    channel=channel,
                    text="Could not extract text from the PDF (may be scanned/image-based)."
                )
                continue

            pdf_chunks = chunk_text(text)
            pdf_name = file_info.get("name", "document.pdf")

            slack_client.chat_postMessage(
                channel=channel,
                text=f"✅ *{pdf_name}* loaded successfully! ({len(pdf_chunks)} chunks)\nNow mention me with a question: `@RAG Bot what is the refund policy?`"
            )

    return jsonify({"ok": True})


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "docs_loaded": pdf_name or "none"})


if __name__ == "__main__":
    print("Starting Slack RAG Bot at http://localhost:8083")
    app.run(debug=True, port=8083)
