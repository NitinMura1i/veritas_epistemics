import os
from dotenv import load_dotenv
import gradio as gr
import requests
from typing import Dict, Optional
from autocorrect import Speller

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
current_article_clean = ""  # Store article content without headers for debate
current_grounding_context = ""  # Store grounding context for debates
original_article = ""  # Store original article before any debates


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
        # Spell-check the topic before searching
        spell = Speller(lang='en')
        corrected_topic = spell(topic)

        # Wikipedia API search endpoint with required headers
        search_url = "https://en.wikipedia.org/w/api.php"
        headers = {
            "User-Agent": "Veritas-Epistemics/1.0 (Educational Project; claude-code@anthropic.com)"
        }

        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": corrected_topic,
            "srlimit": 1
        }
        search_response = requests.get(
            search_url, params=search_params, headers=headers, timeout=10)
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
        content_response = requests.get(
            search_url, params=content_params, headers=headers, timeout=10)
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


def run_multi_agent_debate():
    """Run multi-agent debate on current article."""
    global current_article_clean, current_grounding_context, article_history, original_article

    if not current_article_clean:
        error_msg = "⚠️ No article to debate. Please generate an article first."
        return original_article, error_msg, ""

    # Left panel: original article (before any debates)
    left_display = "📄 ORIGINAL ARTICLE\n"
    left_display += "=" * 44 + "\n\n"
    left_display += original_article

    # Center panel: debate transcript (progressive updates)
    debate_transcript = "🎭 MULTI-AGENT DEBATE\n"
    debate_transcript += "=" * 44 + "\n\n"
    debate_transcript += "⏳ Initializing debate agents...\n\n"

    yield left_display, debate_transcript, ""

    # Agent 1: Defender
    debate_transcript += "🟢 DEFENDER AGENT\n"
    debate_transcript += "-" * 44 + "\n"
    debate_transcript += "⏳ Analyzing article for strengths...\n\n"
    yield left_display, debate_transcript, ""

    try:
        defender_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        defender_chat.append(system(
            "You are the Defender agent in an epistemic debate. Your role is to identify and support "
            "the strongest claims in the article. Provide evidence, reasoning, and citations from the "
            "grounding context. Be rigorous but fair. Provide a concise defense (~200 words)."
        ))
        defender_chat.append(user(
            f"Defend the key claims in this article:\n\n{current_article_clean}\n\nGrounding context:\n{current_grounding_context}"))
        defender_response = defender_chat.sample().content.strip()

        debate_transcript += f"{defender_response}\n\n\n"
        yield left_display, debate_transcript, ""

        # Agent 2: Challenger
        debate_transcript += "🔴 CHALLENGER AGENT\n"
        debate_transcript += "-" * 44 + "\n"
        debate_transcript += "⏳ Analyzing article for weaknesses...\n\n"
        yield left_display, debate_transcript, ""

        challenger_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        challenger_chat.append(system(
            "You are the Challenger agent in an epistemic debate. Your role is to critically examine "
            "the article for weaknesses, overstatements, missing context, or unsupported claims. "
            "Be aggressive but evidence-based. Point out specific issues. Provide a concise critique (~200 words)."
        ))
        challenger_chat.append(user(
            f"Challenge the claims in this article:\n\n{current_article_clean}\n\nGrounding context:\n{current_grounding_context}"))
        challenger_response = challenger_chat.sample().content.strip()

        debate_transcript += f"{challenger_response}\n\n\n"
        yield left_display, debate_transcript, ""

        # Agent 3: Arbiter (produces revised article)
        debate_transcript += "⚖️ ARBITER AGENT\n"
        debate_transcript += "-" * 44 + "\n"
        debate_transcript += "⏳ Synthesizing debate and producing revised article...\n\n"
        yield left_display, debate_transcript, ""

        arbiter_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        arbiter_chat.append(system(
            "You are the Arbiter agent. Read the original article, the Defender's support, and the Challenger's criticisms. "
            "Produce a revised article that:\n"
            "- Strengthens valid points raised by the Defender\n"
            "- Addresses valid criticisms from the Challenger\n"
            "- Adds qualifiers and nuance where needed\n"
            "- Maintains factual accuracy and epistemic integrity\n"
            "- Keeps the same structure and ~300 word length\n"
            "Output ONLY the final revised article in markdown format, with inline citations and a Sources section."
        ))
        arbiter_chat.append(user(f"""Original article:
{current_article_clean}

Defender's arguments:
{defender_response}

Challenger's criticisms:
{challenger_response}

Grounding context:
{current_grounding_context}

Produce the final revised article."""))

        revised_article = arbiter_chat.sample().content.strip()

        debate_transcript += "✅ Debate complete! Revised article generated.\n\n"
        debate_transcript += "**Key improvements:**\n"
        debate_transcript += "- Incorporated Defender's supporting evidence\n"
        debate_transcript += "- Addressed Challenger's valid criticisms\n"
        debate_transcript += "- Added epistemic qualifiers where appropriate\n"

        # Right panel: revised article
        right_display = "📝 REVISED ARTICLE\n"
        right_display += "=" * 44 + "\n\n"
        right_display += revised_article

        # Update current article to the revision for iterative debates
        current_article_clean = revised_article

        # Store revised article in history
        final_article = f"📝 REVISED ARTICLE (Post-Debate)\n"
        final_article += "=" * 44 + "\n\n"
        final_article += f"__Epistemically refined through multi-agent debate__\n\n---\n\n{revised_article}"
        article_history.append(final_article)

        yield left_display, debate_transcript, right_display

    except Exception as e:
        error_transcript = debate_transcript + \
            f"\n\n❌ Error during debate: {str(e)}\n\nPlease try again."
        yield left_display, error_transcript, ""


def fetch_arxiv(topic: str) -> Optional[Dict[str, str]]:
    """Fetch arXiv paper for a given topic."""
    try:
        import xml.etree.ElementTree as ET

        # arXiv API endpoint
        search_url = "http://export.arxiv.org/api/query"

        params = {
            "search_query": f"all:{topic}",
            "start": 0,
            "max_results": 1,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        response = requests.get(search_url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        # Parse XML response
        root = ET.fromstring(response.content)

        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        # Find first entry
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None

        # Extract data
        title = entry.find('atom:title', ns)
        title = title.text.strip().replace('\n', ' ') if title is not None else ""

        summary = entry.find('atom:summary', ns)
        summary = summary.text.strip().replace(
            '\n', ' ') if summary is not None else "No abstract available"

        # Get authors
        authors = []
        for author in entry.findall('atom:author', ns)[:3]:
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)
        authors_str = ", ".join(authors) if authors else "Unknown"
        if len(entry.findall('atom:author', ns)) > 3:
            authors_str += " et al."

        # Get URL
        link = entry.find('atom:id', ns)
        paper_url = link.text if link is not None else ""

        return {
            "title": title,
            "authors": authors_str,
            "content": summary,
            "url": paper_url
        }
    except Exception as e:
        print(f"Error fetching arXiv: {e}")
        return None


def generate_initial_article(topic: str):
    """Generate initial article with Wikipedia and arXiv grounding."""
    global article_history, current_sources, current_article_clean, current_grounding_context, original_article

    # Status in left panel
    status_log = "🔍 PROCESS LOG\n"
    status_log += "=" * 44 + "\n\n"
    status_log += "⏳ Searching Wikipedia for relevant content...\n\n"
    yield "", status_log, ""  # (center, left, right)

    # Fetch Wikipedia
    wiki_data = fetch_wikipedia(topic)

    if wiki_data:
        status_log += f"✅ Found Wikipedia article: '{wiki_data['title']}'\n\n"
    else:
        status_log += f"⚠️ No Wikipedia article found for '{topic}'\n\n"

    # Search arXiv
    status_log += "⏳ Searching arXiv for relevant papers...\n\n"
    yield "", status_log, ""  # (center, left, right)

    paper_data = fetch_arxiv(topic)

    if paper_data:
        status_log += f"✅ Found paper: '{paper_data['title'][:60]}...'\n\n"
    else:
        status_log += f"⚠️ No papers found for '{topic}'\n\n"

    # Build source display for right panel
    sources_display = ""
    source_context = ""
    source_notes = []
    current_sources = []

    if wiki_data and paper_data:
        # Both sources found
        sources_display = "📚 WIKIPEDIA & ARXIV ARTICLES\n"
        sources_display += "=" * 44 + "\n\n"

        sources_display += "WIKIPEDIA:\n"
        sources_display += "-" * 44 + "\n\n"
        sources_display += f"**{wiki_data['title']}**\n\n"
        sources_display += f"Source: {wiki_data['url']}\n\n"
        sources_display += f"{wiki_data['content'][:2500]}\n\n\n"

        sources_display += "ARXIV:\n"
        sources_display += "-" * 44 + "\n\n"
        sources_display += f"**{paper_data['title']}**\n\n"
        sources_display += f"Authors: {paper_data['authors']}\n\n"
        sources_display += f"Source: {paper_data['url']}\n\n"
        sources_display += f"{paper_data['content'][:2500]}"

        source_context = f"Wikipedia Article: {wiki_data['title']}\n\n{wiki_data['content'][:2000]}\n\n---\n\narXiv Paper: {paper_data['title']}\nAuthors: {paper_data['authors']}\n\n{paper_data['content'][:2000]}"
        source_notes.append(
            f"[Wikipedia: {wiki_data['title']}]({wiki_data['url']})")
        source_notes.append(
            f"[arXiv: {paper_data['title'][:50]}...]({paper_data['url']})")

        current_sources = [
            {"name": wiki_data['title'], "type": "Wikipedia",
                "url": wiki_data['url'], "content": wiki_data['content']},
            {"name": paper_data['title'], "type": "arXiv", "url": paper_data['url'],
                "content": paper_data['content'], "authors": paper_data['authors']}
        ]

    elif wiki_data:
        # Only Wikipedia found
        sources_display = "📚 WIKIPEDIA ARTICLE\n"
        sources_display += "=" * 44 + "\n\n"
        sources_display += f"**{wiki_data['title']}**\n\n"
        sources_display += f"Source: {wiki_data['url']}\n\n"
        sources_display += "---\n\n"
        sources_display += f"{wiki_data['content'][:2500]}"

        source_context = f"Wikipedia Article: {wiki_data['title']}\n\n{wiki_data['content'][:3000]}"
        source_notes.append(
            f"[Wikipedia: {wiki_data['title']}]({wiki_data['url']})")
        current_sources = [{"name": wiki_data['title'], "type": "Wikipedia",
                            "url": wiki_data['url'], "content": wiki_data['content']}]

    elif paper_data:
        # Only Semantic Scholar found
        sources_display = "📚 ARXIV ARTICLE\n"
        sources_display += "=" * 44 + "\n\n"
        sources_display += f"**{paper_data['title']}**\n\n"
        sources_display += f"Authors: {paper_data['authors']}\n\n"
        sources_display += f"Source: {paper_data['url']}\n\n"
        sources_display += "---\n\n"
        sources_display += f"{paper_data['content'][:2500]}"

        source_context = f"arXiv Paper: {paper_data['title']}\nAuthors: {paper_data['authors']}\n\n{paper_data['content'][:3000]}"
        source_notes.append(
            f"[arXiv: {paper_data['title'][:50]}...]({paper_data['url']})")
        current_sources = [{"name": paper_data['title'], "type": "arXiv", "url": paper_data['url'],
                            "content": paper_data['content'], "authors": paper_data['authors']}]

    else:
        # No sources found
        source_context = f"No specific source found. Generating article about: {topic}"
        current_sources = []

    # Show sources in right panel - keep placeholder during generation
    loading_message = "📚 SOURCE MATERIAL\n" + "=" * 44 + \
        "\n\n⏳ Loading sources...\n\nSources will appear here when article generation is complete."

    if current_sources:
        status_log += "⏳ Generating epistemically grounded article...\n\n"
    else:
        status_log += "⏳ Generating article from general knowledge (ungrounded)...\n\n"
    yield "", status_log, loading_message  # (center, left, right)

    # Build source note for article
    if source_notes:
        source_note = f"Grounded in: {' + '.join(source_notes)}"
    else:
        source_note = "⚠️ No sources found - generated from general knowledge"

    # Generate article with xAI
    source_instruction = ""
    if wiki_data and paper_data:
        source_instruction = f"Use the Wikipedia article to define the scope and structure of your article about '{topic}'. Use the arXiv paper ONLY to add academic depth where directly relevant, not to shift focus to niche sub-topics or specific research questions in the paper."
    elif wiki_data:
        source_instruction = "Use the following Wikipedia content as your primary source:"
    elif paper_data:
        source_instruction = f"Use the following arXiv paper as your primary source. Keep the article broadly focused on '{topic}' as a general topic rather than narrowly focused on the specific research question in the paper."
    else:
        source_instruction = "Generate based on your knowledge:"

    prompt = f"""You are an expert knowledge synthesizer. Write a comprehensive, factual article about "{topic}".

{source_instruction}

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

        # Add header and source note at the top
        final_article = f"📝 YOUR ARTICLE\n"
        final_article += "=" * 44 + "\n\n"
        final_article += f"__{source_note}__\n\n---\n\n{article_content}"

        # Store in history
        article_history.append(final_article)

        # Store clean article and grounding context for debates
        current_article_clean = article_content
        original_article = article_content
        current_grounding_context = source_context

        # Update status log to show completion
        status_log += "✅ Article generation complete!\n\n"

        # Rebuild sources display for final yield
        final_sources_display = ""
        if len(current_sources) == 2:
            # Both Wikipedia and Semantic Scholar
            final_sources_display = "📚 WIKIPEDIA & ARXIV ARTICLES\n"
            final_sources_display += "=" * 44 + "\n\n"

            final_sources_display += "WIKIPEDIA:\n"
            final_sources_display += "-" * 44 + "\n\n"
            final_sources_display += f"**{current_sources[0]['name']}**\n\n"
            final_sources_display += f"Source: {current_sources[0]['url']}\n\n"
            final_sources_display += f"{current_sources[0]['content'][:2500]}\n\n\n"

            final_sources_display += "ARXIV:\n"
            final_sources_display += "-" * 44 + "\n\n"
            final_sources_display += f"**{current_sources[1]['name']}**\n\n"
            final_sources_display += f"Authors: {current_sources[1]['authors']}\n\n"
            final_sources_display += f"Source: {current_sources[1]['url']}\n\n"
            final_sources_display += f"{current_sources[1]['content'][:2500]}"

        elif len(current_sources) == 1:
            # Single source
            source = current_sources[0]
            if source['type'] == 'Wikipedia':
                final_sources_display = "📚 WIKIPEDIA ARTICLE\n"
                final_sources_display += "=" * 44 + "\n\n"
                final_sources_display += f"**{source['name']}**\n\n"
                final_sources_display += f"Source: {source['url']}\n\n"
                final_sources_display += "---\n\n"
                final_sources_display += f"{source['content'][:2500]}"
            else:  # Semantic Scholar
                final_sources_display = "📚 ARXIV ARTICLE\n"
                final_sources_display += "=" * 44 + "\n\n"
                final_sources_display += f"**{source['name']}**\n\n"
                final_sources_display += f"Authors: {source['authors']}\n\n"
                final_sources_display += f"Source: {source['url']}\n\n"
                final_sources_display += "---\n\n"
                final_sources_display += f"{source['content'][:2500]}"

        # Yield: center (article), left (status), right (sources)
        yield final_article, status_log, final_sources_display

    except Exception as e:
        status_log += f"❌ Error generating article: {str(e)}\n\nPlease try again.\n"
        yield "", status_log, ""


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
        html, body {
            overflow-x: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
            max-height: 100vh !important;
            overflow-y: auto !important;
        }
        body > div,
        .gradio-container,
        .gradio-container > div,
        #root,
        .app,
        .main,
        .wrap,
        .block {
            padding-bottom: 0px !important;
            margin-bottom: 0px !important;
            min-height: unset !important;
        }
        /* Force cut off after content */
        .gradio-container::after {
            content: '' !important;
            display: block !important;
            height: 0px !important;
            clear: both !important;
        }
        .chat-input-container {
            position: relative;
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
            width: 32px !important;
            height: 32px !important;
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

        /* Action button styling */
        #action-button {
            background-color: #0f0f0f !important;
            border: 2px solid #6366f1 !important;
            color: #6366f1 !important;
            border-radius: 8px !important;
            font-family: monospace !important;
            font-size: 0.95rem !important;
            font-weight: normal !important;
            padding: 6px 20px !important;
            height: 38px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }

        #action-button:hover {
            background-color: #1a1a1a !important;
            border-color: #4f46e5 !important;
            color: #4f46e5 !important;
        }

        #action-button:disabled {
            opacity: 0.4 !important;
            cursor: not-allowed !important;
            border-color: #444444 !important;
            color: #666666 !important;
        }

        #action-button:disabled:hover {
            background-color: #0f0f0f !important;
            border-color: #444444 !important;
            color: #666666 !important;
        }
        /* Top control row - dropdown and input */
        .top-control-row {
            max-width: 1200px !important;
            margin: 20px auto !important;
            gap: 15px !important;
            align-items: center !important;
        }

        /* Dropdown styling - comprehensive targeting */
        label[id*="component"] select,
        select[class*="dropdown"],
        .gr-dropdown,
        .gr-box select {
            background-color: #111111 !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
            border-radius: 8px !important;
            font-family: monospace !important;
            font-size: 1rem !important;
        }

        /* Dropdown input field */
        .gr-dropdown input,
        input[role="combobox"] {
            background-color: #111111 !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
            font-family: monospace !important;
            font-size: 1rem !important;
        }

        /* Dropdown container */
        .gr-dropdown,
        div[class*="dropdown"] {
            background-color: #111111 !important;
        }

        /* Dropdown options list */
        ul[role="listbox"],
        div[role="listbox"],
        .options,
        [class*="options"] {
            background-color: #1a1a1a !important;
            border: 1px solid #444444 !important;
            border-radius: 8px !important;
        }

        /* Individual dropdown options */
        li[role="option"],
        div[role="option"],
        .option,
        [class*="option"]:not([class*="options"]) {
            background-color: #1a1a1a !important;
            color: #e5e7eb !important;
            padding: 8px 12px !important;
            font-family: monospace !important;
            font-size: 0.9rem !important;
        }

        /* Dropdown option hover state */
        li[role="option"]:hover,
        div[role="option"]:hover,
        .option:hover,
        li[aria-selected="true"],
        div[aria-selected="true"] {
            background-color: #6366f1 !important;
            color: white !important;
        }

        /* Epistemic dropdown styling - MUST come LAST to override all above */
        .epistemic-dropdown-borderless input[role="combobox"],
        .epistemic-dropdown-borderless input,
        .epistemic-dropdown-borderless span,
        .epistemic-dropdown-borderless div {
            color: #e5e7eb !important;
            font-family: monospace !important;
            font-size: 0.9rem !important;
        }

        .epistemic-dropdown-borderless input[role="combobox"] {
            padding: 14px 18px !important;
            background-color: #111111 !important;
            border: 1px solid #444444 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
            line-height: 1.5 !important;
        }

        .epistemic-dropdown-borderless input[role="combobox"]::placeholder {
            color: #aaaaaa !important;
            opacity: 0.8 !important;
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
            margin: -20px auto 0px auto !important;
            gap: 20px !important;
            padding: 0 20px 0 20px !important;
            align-items: flex-start !important;
            border: none !important;
            background: transparent !important;
        }

        /* Kill any space after article row */
        .article-row ~ * {
            display: none !important;
        }

        /* Ensure container height fits content */
        .gradio-container {
            max-height: fit-content !important;
        }

        /* Force all columns in article row to start at same height */
        .article-row > div[class*="column"] {
            align-self: flex-start !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        .article-row > div,
        .article-row div[class*="block"],
        .article-row div[class*="container"],
        .article-row div[class*="wrap"],
        .article-row > * > div,
        .article-row label {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        /* Ensure all textboxes start at the same vertical position */
        .article-row textarea {
            margin-top: 0 !important;
            vertical-align: top !important;
        }

        .article-row label {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        /* Nuclear option: force ALL elements containing our panels to align at top */
        .article-row *:has(.side-panel),
        .article-row *:has(.left-panel),
        .article-row *:has(.right-panel),
        .article-row *:has(.central-article) {
            margin-top: 0 !important;
            padding-top: 0 !important;
            align-self: flex-start !important;
        }

        /* Force the direct containers of each panel */
        .side-panel,
        .left-panel,
        .right-panel {
            margin-top: 0 !important;
        }

        /* Side panels (left and right commentary) */
        /* All panels always visible with consistent sizing */
        .side-panel {
            border-radius: 12px !important;
            background-color: #1a1a1a !important;
            color: #e5e7eb !important;
            border: 4px solid #444444 !important;
            padding: 16px !important;
            font-family: monospace !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
            min-height: 2000px !important;
            max-height: 2000px !important;
            overflow-y: auto !important;
            box-shadow: 0 2px 12px rgba(99, 102, 241, 0.1) !important;
            transition: all 0.3s ease !important;
            margin-top: 0 !important;
        }

        .left-panel {
            border-left: 4px solid #10b981 !important;
            margin-top: 1px !important;
        }

        .right-panel {
            border-right: 4px solid #f59e0b !important;
            margin-top: 1px !important;
        }

        /* Central article styling */
        .central-article {
            border-radius: 12px !important;
            background-color: #1a1a1a !important;
            color: #e5e7eb !important;
            border: 4px solid #6366f1 !important;
            padding: 16px !important;
            font-family: monospace !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
            min-height: 2000px !important;
            max-height: 2000px !important;
            overflow-y: auto !important;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
            margin-top: 0 !important;
        }

        /* Force central article label to have no top offset */
        label.central-article {
            margin-top: 0 !important;
        }

        /* Override to force textarea borders */
        .side-panel textarea,
        .left-panel textarea,
        .right-panel textarea,
        .central-article textarea {
            border: 1px solid #ffffff !important;
            border-radius: 12px !important;
        }
        .central-article::placeholder {
            color: #777777 !important;
            opacity: 0.7 !important;
            font-style: italic !important;
            text-align: center !important;
        }
    </style>
    <script>
        // Scroll textareas to top after content updates
        function scrollToTop() {
            const textareas = document.querySelectorAll('.central-article textarea, .side-panel textarea');
            textareas.forEach(textarea => {
                textarea.scrollTop = 0;
            });
        }

        // Run on page load and periodically check for updates
        window.addEventListener('load', scrollToTop);
        setInterval(scrollToTop, 500);
    </script>
    <style>
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

    # Top control row: dropdown and input box
    with gr.Row(elem_classes=["top-control-row"]):
        # Epistemic tools dropdown
        epistemic_dropdown = gr.Dropdown(
            choices=[
                "Article Generation",
                "Multi-Agent Debate",
                "Self-Critique",
                "Synthetic Data",
                "Epistemic Score",
                "Update Simulation",
                "User Feedback"
            ],
            show_label=False,
            value="Article Generation",
            interactive=True,
            scale=0,
            min_width=200,
            elem_classes=["epistemic-dropdown-borderless"],
            container=False
        )

        # Chat-style input with send arrow
        with gr.Column(elem_classes=["chat-input-container"], scale=1):
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

        # Dynamic action button
        action_button = gr.Button(
            value="Generate Article",
            variant="primary",
            scale=0,
            min_width=180,
            elem_id="action-button"
        )

    # The article panels row - now full width!
    # All 3 panels always visible to keep center article perfectly centered
    with gr.Row(elem_classes=["article-row"]):
        # Left panel - Process log and status
        left_panel = gr.Textbox(
            value="🔍 PROCESS LOG\n" + "=" * 44 + "\n\nThis panel displays real-time status updates during article generation:\n\n- Wikipedia search progress\n- arXiv paper search progress\n- Source retrieval status\n- Article generation steps\n- Completion notifications\n\nEnter a topic and click the arrow to begin!",
            lines=60,
            interactive=False,
            show_copy_button=False,
            show_label=False,
            container=True,
            elem_classes=["side-panel", "left-panel"],
            visible=True
        )

        # Central article (always visible and centered!)
        article_display = gr.Textbox(
            value="📝 YOUR ARTICLE\n" + "=" * 44 +
            "\n\nYour generated article will appear here.\n\nIterate it in order to get as close to the truth as you can!",
            lines=60,
            interactive=False,
            show_copy_button=False,
            container=True,
            elem_classes=["side-panel", "central-article"],
            show_label=False
        )

        # Right panel - Source material (Wikipedia & arXiv)
        right_panel = gr.Textbox(
            value="📚 SOURCE MATERIAL\n" + "=" * 44 +
            "\n\nThis panel displays source articles used to generate your article:\n\n- Wikipedia articles (general knowledge)\n- arXiv papers (academic research)\n- Source URLs and titles\n- Reference material for verification\n\nSources will appear here after article generation.",
            lines=60,
            interactive=False,
            show_copy_button=False,
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
    ).then(
        fn=None,
        js="""() => {
            setTimeout(() => {
                const textareas = document.querySelectorAll('textarea');
                textareas.forEach(t => { t.scrollTop = 0; });
            }, 100);
        }"""
    )

    # Also trigger on Enter key in topic input
    topic_input.submit(
        fn=generate_initial_article,
        inputs=[topic_input],
        outputs=[article_display, left_panel, right_panel]
    ).then(
        fn=None,
        js="""() => {
            setTimeout(() => {
                const textareas = document.querySelectorAll('textarea');
                textareas.forEach(t => { t.scrollTop = 0; });
            }, 100);
        }"""
    )

    # Update action button text and input state based on dropdown selection
    def update_ui_state(selected_tool):
        button_text_map = {
            "Article Generation": "Generate Article",
            "Multi-Agent Debate": "Start Debate",
            "Self-Critique": "Critique Article",
            "Synthetic Data": "Generate Data",
            "Epistemic Score": "Score Article",
            "Update Simulation": "Simulate Updates",
            "User Feedback": "Collect Feedback"
        }

        button_text = button_text_map.get(selected_tool, "Generate Article")

        # Disable topic input when epistemic tool is selected
        topic_disabled = (selected_tool != "Article Generation")

        # Check if article exists for epistemic tools
        button_disabled = topic_disabled and not current_article_clean

        return (
            gr.update(value=button_text, interactive=not button_disabled),
            gr.update(interactive=not topic_disabled)
        )

    epistemic_dropdown.change(
        fn=update_ui_state,
        inputs=[epistemic_dropdown],
        outputs=[action_button, topic_input]
    )

    # Route action button to correct function based on dropdown
    def execute_action(selected_tool, topic):
        if selected_tool == "Article Generation":
            # Generate article
            yield from generate_initial_article(topic)
        elif selected_tool == "Multi-Agent Debate":
            # Run debate
            yield from run_multi_agent_debate()
        else:
            # Placeholder for other tools
            error_msg = f"⚠️ {selected_tool} not yet implemented."
            yield "", error_msg, ""

    action_button.click(
        fn=execute_action,
        inputs=[epistemic_dropdown, topic_input],
        outputs=[article_display, left_panel, right_panel]
    ).then(
        fn=None,
        js="""() => {
            setTimeout(() => {
                const textareas = document.querySelectorAll('textarea');
                textareas.forEach(t => { t.scrollTop = 0; });
            }, 100);
        }"""
    )


# Launch without footer
demo.launch(show_api=False)
