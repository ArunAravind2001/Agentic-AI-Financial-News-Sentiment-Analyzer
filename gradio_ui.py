import gradio as gr
from first_agentic_ai import graph  

def analyze_news(query):
    try:
        result = graph.invoke({"query": query})
        companies = result.get("top_companies", [])
        if not companies:
            return "⚠️ No bullish companies found for this topic."
        return "\n".join([f"🔹 {c}" for c in companies])
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Build the Gradio interface
iface = gr.Interface(
    fn=analyze_news,
    inputs=gr.Textbox(label="Enter news topic (e.g., tech, finance, expansion)"),
    outputs=gr.Textbox(label="Top Bullish Companies", lines=10),
    title="🧠 Agentic AI: News Sentiment Analyzer",
    description="Enter a topic to analyze recent news sentiment and discover top bullish companies."
)

if __name__ == "__main__":
    iface.launch()
