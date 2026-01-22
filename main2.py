import os
from dotenv import load_dotenv
import gradio as gr
import requests
from typing import Dict, Optional

from xai_sdk import Client
from xai_sdk.chat import user, system

load_dotenv()

api_key = os.getenv("XAI_API_KEY")
if api_key is None:
    raise ValueError("XAI_API_KEY not found in .env file!")

client = Client(api_key=api_key, timeout=3600)

# State management
article_history = []
current_sources = []
source_visible = False  # Track if source panel is visible

def toggle_source_view():
    """Toggle Wikipedia source panel content."""
    global source_visible, current_sources

    source_visible = not source_visible

    if source_visible and current_sources:
        # Show Wikipedia source content
        source = current_sources[0]
        wiki_display = f"# 📚 Source Material\n\n**{source['name']}**\n\n{source.get('url', '')}\n\n---\n\n{source.get('content', '')[:2000]}..."
        return wiki_display
    else:
        # Return to default placeholder text
        return "# 📚 Source Material\n\nClick **'View Source'** to see the Wikipedia article or other source material used to generate the article.\n\nThis panel will also display retrieved context and grounding information during article generation."

def fetch_wikipedia(topic: str) -> Optional[Dict[str, str]]:
    """Fetch Wikipedia article content for a given topic."""
    try:
        # Wikipedia API search endpoint with required headers
        search_url = "https://en.wikipedia.org/w/api.php"
        headers = {
            "User-Agent": "Veritas-Epistemics/1.0 (Educational Project; claude-code@anthropic.com)"
        }

        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": topic,
            "srlimit": 1
        }
        search_response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        search_data = search_response.json()

        if not search_data.get("query", {}).get("search"):
            return None

        # Get the page title
        page_title = search_data["query"]["search"][0]["title"]

        # Fetch page content
        content_params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "extracts",
            "explaintext": True
        }
        content_response = requests.get(search_url, params=content_params, headers=headers, timeout=10)
        content_data = content_response.json()

        pages = content_data.get("query", {}).get("pages", {})
        page_id = list(pages.keys())[0]
        extract = pages[page_id].get("extract", "")

        return {
            "title": page_title,
            "content": extract,
            "url": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        }
    except Exception as e:
        print(f"Error fetching Wikipedia: {e}")
        return None

def generate_initial_article(topic: str):
    """Generate initial article with Wikipedia grounding."""
    global article_history, current_sources

    # Progressive update 1: Fetching Wikipedia - keep center centered!
    yield "🔍 Searching Wikipedia for relevant content...", "", ""

    # Fetch Wikipedia
    wiki_data = fetch_wikipedia(topic)

    if wiki_data:
        # Don't auto-show Wikipedia - keeps center article centered
        yield f"📚 Found Wikipedia article: '{wiki_data['title']}'\n\n✍️ Generating epistemically grounded article...", "", ""

        source_context = f"Wikipedia Article: {wiki_data['title']}\n\n{wiki_data['content'][:3000]}"  # Use first 3000 chars
        source_note = f"Grounded in Wikipedia article: [{wiki_data['title']}]({wiki_data['url']})"
        current_sources = [{
            "name": wiki_data['title'],
            "type": "Wikipedia",
            "url": wiki_data['url'],
            "content": wiki_data['content']
        }]
    else:
        yield f"⚠️ No Wikipedia article found for '{topic}'\n\n✍️ Generating article from general knowledge (ungrounded)...", "", ""
        source_context = f"No specific source found. Generating article about: {topic}"
        source_note = "⚠️ No Wikipedia source found - generated from general knowledge"
        current_sources = []

    # Generate article with xAI
    prompt = f"""You are an expert knowledge synthesizer. Write a comprehensive, factual article about "{topic}".

{"Use the following Wikipedia content as your primary source:" if wiki_data else "Generate based on your knowledge:"}

{source_context}

Requirements:
1. Write in encyclopedic style (factual, neutral, well-structured)
2. Include inline citations like [1] when referencing facts
3. Add a "## Sources" section at the end listing all references
4. Be clear about certainty levels - use phrases like "evidence suggests", "widely accepted", etc.
5. Length: EXACTLY 300 words (excluding the Sources section)
6. Structure: Introduction, 2-3 main sections, conclusion, sources

Format the article in clean markdown."""

    try:
        # xAI SDK chat - create chat object and append messages
        chat = client.chat.create(model="grok-4-1-fast-reasoning")

        chat.append(system(
            "You are an expert knowledge synthesizer focused on epistemic integrity. "
            "Write factual, well-sourced articles with inline citations. "
            "Be clear about certainty levels and avoid speculation."
        ))

        chat.append(user(prompt))

        response = chat.sample()
        article_content = response.content.strip()

        # Add source note at the top
        final_article = f"__{source_note}__\n\n---\n\n{article_content}"

        # Store in history
        article_history.append(final_article)

        # Keep side panels empty - center stays centered!
        yield final_article, "", ""

    except Exception as e:
        error_msg = f"❌ Error generating article: {str(e)}\n\nPlease try again."
        yield error_msg, "", ""

# Dark theme
dark_theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#0f0f0f",
    body_background_fill_dark="#0f0f0f",
    block_background_fill="#1a1a1a",
    block_background_fill_dark="#1a1a1a",
    block_title_text_color="#ffffff",
    block_label_text_color="#e0e0e0",
    input_background_fill="#111111",
    input_background_fill_dark="#111111",
    input_border_color="#444444",
    input_border_color_dark="#444444",
    input_border_color_focus="#6366f1",
    body_text_color="#e5e7eb",
    body_text_color_dark="#e5e7eb",
    body_text_size="md",
)

with gr.Blocks(theme=dark_theme, title="Veritas Epistemics - Truth-Seeking Article Generator") as demo:
    # Centered title image
    gr.Image(
        value="veritas_title.png",
        show_label=False,
        interactive=False,
        container=False,
        height=120,
        width=None,
        elem_id="veritas-title",
        show_download_button=False,
        show_fullscreen_button=False,
    )

    # CSS: center title + clean chat-style input with arrow
    gr.HTML("""
    <style>
        #veritas-title {
            text-align: center !important;
            margin: 30px auto -50px auto !important;
            padding: 0 !important;
        }
        #veritas-title img {
            display: block !important;
            margin: 0 auto !important;
            max-width: 90% !important;
        }
        footer {
            display: none !important;  /* hide Gradio footer */
        }
        .chat-input-container {
            position: relative;
            max-width: 800px !important;
            margin: 0 auto 20px auto !important;
        }
        .chat-input-container .gr-textbox {
            border-radius: 12px !important;
            padding: 14px 70px 14px 18px !important;
            background-color: #111111 !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        }
        .chat-input-container .gr-textbox textarea {
            resize: none !important;
            font-size: 1.1rem !important;
            font-family: monospace !important;
            line-height: 1.5 !important;
        }
        .chat-input-container .gr-textbox textarea::placeholder {
            color: #aaaaaa !important;
            opacity: 0.8 !important;
            font-size: 1rem !important;
            font-family: monospace !important;
        }
        #topic-input-box {
            font-family: monospace !important;  /* change to any font you want */
            font-size: 1.1rem !important;  /* optional: keep or adjust size */
        }

        #topic-input-box textarea {
            font-family: monospace !important;  /* main typed text */
        }

        #topic-input-box::placeholder {
            font-family: monospace !important;  /* placeholder text */
            font-size: 0.7rem !important;  /* your existing small size */
        }
        .send-btn {
            position: absolute !important;
            bottom: 3px !important;
            right: 3px !important;
            width: 36px !important;
            height: 36px !important;
            border-radius: 8px !important;
            background-color: #6366f1 !important;
            color: white !important;
            border: none !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.6rem !important;
            z-index: 10 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
        }
        .send-btn:hover {
            background-color: #4f46e5 !important;
        }
        /* Action buttons row */
        .action-buttons-row {
            max-width: 1200px !important;
            margin: 30px auto 20px auto !important;
            gap: 12px !important;
            display: flex !important;
            justify-content: center !important;
            flex-wrap: wrap !important;
        }

        .action-buttons-row button {
            flex: 1 1 180px !important;
            max-width: 200px !important;
            min-width: 160px !important;
        }

        /* Utility buttons row */
        .utility-buttons-row {
            display: flex !important;
            justify-content: center !important;
            gap: 12px !important;
            margin: 10px auto 20px auto !important;
            max-width: 500px !important;
        }

        .utility-btn {
            background-color: #2a2a2a !important;
            border: 1px solid #6366f1 !important;
            color: #6366f1 !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            min-width: 160px !important;
        }

        .utility-btn:hover {
            background-color: #6366f1 !important;
            color: white !important;
        }

        /* Article row - contains side panels + central article */
        .article-row {
            max-width: 1600px !important;
            margin: 20px auto !important;
            gap: 20px !important;
            padding: 0 20px !important;
            align-items: flex-start !important;
        }

        /* Side panels (left and right commentary) */
        /* All panels always visible with consistent sizing */
        .side-panel {
            border-radius: 12px !important;
            background-color: #1a1a1a !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
            padding: 16px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
            min-height: 400px !important;
            max-height: 600px !important;
            overflow-y: auto !important;
            box-shadow: 0 2px 12px rgba(99, 102, 241, 0.1) !important;
            transition: all 0.3s ease !important;
        }

        .left-panel {
            border-left: 3px solid #10b981 !important;
        }

        .right-panel {
            border-right: 3px solid #f59e0b !important;
        }

        /* Central article styling */
        .central-article {
            border-radius: 12px !important;
            background-color: #1a1a1a !important;
            color: #e5e7eb !important;
            border: 1px solid #6366f1 !important;
            border-width: 2px !important;
            padding: 20px !important;
            font-family: monospace !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
            min-height: 400px !important;
            max-height: 600px !important;
            overflow-y: auto !important;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
        }
        .central-article::placeholder {
            color: #777777 !important;
            opacity: 0.7 !important;
            font-style: italic !important;
            text-align: center !important;
        }
        /* Toolbar nuke (your existing rule kept) */
        .icon-button-wrapper.top-panel.hide-top-corner,
        .icon-button-wrapper.top-panel,
        .icon-button-wrapper,
        .top-panel,
        .hide-top-corner,
        div[class*="icon-button-wrapper"],
        div[class*="top-panel"],
        div[class*="hide-top-corner"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        .gr-image-container:has(div.icon-button-wrapper),
        .gr-image-container:has(.top-panel),
        .gr-image-container:has(.hide-top-corner) {
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }
    </style>
    """)

    # Chat-style input with send arrow in corner
    with gr.Column(elem_classes=["chat-input-container"]):
        topic_input = gr.Textbox(
            placeholder="Enter a topic of your choice. e.g. Machine Learning, Astrology, Global Warming",
            lines=1,
            container=False,
            elem_id="topic-input-box"
        )

        send_arrow = gr.Button(
            value="➤",
            variant="primary",
            elem_id="send-arrow-btn",
            elem_classes=["send-btn"],
            size="sm"
        )

    # Action buttons row (all at top)
    with gr.Row(elem_classes=["action-buttons-row"]):
        multi_agent_btn = gr.Button("🔄 Multi-Agent Debate", size="lg", variant="secondary")
        critique_btn = gr.Button("🔍 Self-Critique", size="lg", variant="secondary")
        synthetic_btn = gr.Button("🧪 Synthetic Data", size="lg", variant="secondary")
        scorecard_btn = gr.Button("📊 Epistemic Score", size="lg", variant="secondary")
        update_btn = gr.Button("🔄 Update Simulation", size="lg", variant="secondary")
        feedback_btn = gr.Button("💬 User Feedback", size="lg", variant="secondary")

    # Utility buttons row (centered)
    with gr.Row(elem_classes=["utility-buttons-row"]):
        view_source_btn = gr.Button("📚 View Source", size="sm", elem_classes=["utility-btn"])
        compare_btn = gr.Button("↕️ Compare Versions", size="sm", elem_classes=["utility-btn"])

    # The article panels row - now full width!
    # All 3 panels always visible to keep center article perfectly centered
    with gr.Row(elem_classes=["article-row"]):
        # Left panel - Source material and context
        left_panel = gr.Textbox(
            value="# 📚 Source Material\n\nClick **'View Source'** to see the Wikipedia article or other source material used to generate the article.\n\nThis panel will also display retrieved context and grounding information during article generation.",
            lines=20,
            interactive=False,
            show_label=False,
            container=True,
            elem_classes=["side-panel", "left-panel"],
            visible=True
        )

        # Central article (always visible and centered!)
        article_display = gr.Textbox(
            value="This is where your article will appear. Iterate it in order to get as close to the truth as you can!",
            lines=20,
            interactive=False,
            show_copy_button=True,
            container=True,
            elem_classes=["central-article"],
            show_label=False
        )

        # Right panel - Agent debates and evaluations
        right_panel = gr.Textbox(
            value="# 🔄 Epistemic Processing\n\nThis panel will display:\n\n- **Multi-agent debates** between advocates and skeptics\n- **Self-critique** reasoning and improvements\n- **Epistemic scoring** and quality metrics\n- **Real-time updates** and simulations\n\nUse the action buttons above to activate different epistemic tools.",
            lines=20,
            interactive=False,
            show_label=False,
            container=True,
            elem_classes=["side-panel", "right-panel"],
            visible=True
        )

    # Wire up the send button
    send_arrow.click(
        fn=generate_initial_article,
        inputs=[topic_input],
        outputs=[article_display, left_panel, right_panel]
    )

    # Also trigger on Enter key in topic input
    topic_input.submit(
        fn=generate_initial_article,
        inputs=[topic_input],
        outputs=[article_display, left_panel, right_panel]
    )

    # Wire up view source button
    view_source_btn.click(
        fn=toggle_source_view,
        inputs=[],
        outputs=[left_panel]
    )

# Launch without footer
demo.launch(show_api=False)
