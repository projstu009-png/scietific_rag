# src/app.py
import gradio as gr
from rag_system import ScientificRAGSystem

# --- System Initialization ---
# Initialize the RAG system (ensure config is correct)
try:
    rag_system = ScientificRAGSystem("config/config.yaml")
    print("Scientific RAG System initialized successfully for Gradio UI.")
except Exception as e:
    print(f"Error initializing RAG system: {e}")
    # Define a dummy system to allow the UI to launch for debugging
    class DummyRAG:
        def query(self, question):
            return {
                'answer': f"Error: RAG system not initialized. {e}",
                'sources': [],
                'answerable': False,
                'retrieval_quality': 0.0,
                'confidence': 0.0
            }
    rag_system = DummyRAG()

# --- Core Functions ---
def answer_question(question: str, history: list):
    """Main function to get an answer from the RAG system"""
    if not question:
        return "Please ask a question.", history, ""

    # Get the agentic response
    result = rag_system.query(question)

    # Format the answer
    answer = result.get('answer', "No answer found.")

    # Format sources for display
    sources = result.get('sources', [])
    formatted_sources = ""
    if sources:
        for i, source in enumerate(sources[:5]):  # Show top 5
            score = source.get('score', 0)
            text = source.get('text', 'N/A').replace('\n', ' ')
            preview = text[:200] + "..." if len(text) > 200 else text
            formatted_sources += f"**[{i+1}] Score: {score:.3f}**\n*\"{preview}\"*\n\n"

    # Update chatbot history
    history.append((question, answer))

    return "", history, formatted_sources

def ingest_files(files: list):
    """Function to handle file uploads for ingestion"""
    if not files:
        return "No files uploaded."

    # Get file paths
    file_paths = [f.name for f in files]

    try:
        # Ingest papers
        rag_system.ingest_papers(file_paths)
        return f"Successfully ingested {len(file_paths)} file(s)!"
    except Exception as e:
        return f"Error during ingestion: {e}"

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Scientific RAG System UI")
    gr.Markdown("Ask questions about your ingested scientific papers. The system uses an agentic workflow to provide answers with citations.")

    with gr.Tab("QA Chat"):
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            height=500,
            avatar_images=(("images/user.png"), "images/bot.png"),
            label="Scientific Chat"
        )
        chat_input = gr.Textbox(
            lines=1,
            label="Ask your question here...",
            placeholder="e.g., What is the role of Xe-135 in nuclear reactors?",
            render=False,
            container=False,
        )

        with gr.Row():
            chat_input.render()
            submit_btn = gr.Button("Submit", variant="primary")

        gr.Markdown("### Sources")
        sources_output = gr.Markdown(label="Retrieved Sources")

    with gr.Tab("Ingest Papers"):
        file_uploader = gr.File(
            label="Upload PDF Papers",
            file_count="multiple",
            file_types=[".pdf"],
            type="file"
        )
        ingest_button = gr.Button("Ingest", variant="primary")
        ingest_status = gr.Textbox(label="Ingestion Status", interactive=False)

    # --- Event Handling ---
    # Chat functionality
    chat_msg = submit_btn.click(
        answer_question,
        [chat_input, chatbot],
        [chat_input, chatbot, sources_output],
        queue=True
    )
    chat_input.submit(
        answer_question,
        [chat_input, chatbot],
        [chat_input, chatbot, sources_output],
        queue=True
    )

    # Ingestion functionality
    ingest_button.click(
        ingest_files,
        [file_uploader],
        [ingest_status],
        queue=True
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
