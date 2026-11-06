import gradio as gr
from backend.main import app as fastapi_app

# Define a clean landing page for Hugging Face Space
with gr.Blocks(title="e-Netra AI Screening Backend") as demo:
    gr.Markdown("# 👁️ e-Netra AI Screening Backend")
    gr.Markdown(
        """
        ### 🟢 Backend Service is Running!
        
        This space hosts the **e-Netra AI FastAPI Backend**. All REST API endpoints and WebSockets are live and active.
        
        * 📖 **Interactive Swagger UI**: [Open API Docs (/docs)](/docs)
        * 🩺 **System Health Check**: [Check Health Status (/health)](/health)
        * 🌐 **API Base URL for Frontend**: Connect your React / Flutter app directly to this space URL.
        """
    )

# Mount Gradio onto the existing FastAPI app
# This preserves all FastAPI routes (/docs, /health, /api/..., and /ws/...)
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

