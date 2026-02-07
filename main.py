import os
from dotenv import load_dotenv
import gradio as gr
import requests
from typing import Dict, Optional
from autocorrect import Speller

from xai_sdk import Client
from xai_sdk.chat import user, system
from xai_sdk.tools import web_search

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
version_panel_visible = False  # Track if version history panel is visible


def toggle_source_view():
    """Toggle source panel content."""
    global source_visible, current_sources

    source_visible = not source_visible


def toggle_version_panel():
    """Toggle version history panel visibility."""
    import json
    global version_panel_visible
    version_panel_visible = not version_panel_visible
    return json.dumps(build_version_history_html())


def build_version_history_html():
    """Build HTML for version history list with article content as data attributes."""
    global article_history

    if not article_history:
        return {
            "list": "<div style='color: #9ca3af; text-align: center; padding: 20px; font-size: 0.65rem;'>No versions yet.</div>",
            "articles": [],
            "latest_content": "No versions yet. Generate an article to begin."
        }

    list_html = ""
    articles_data = []

    for idx, article in enumerate(reversed(article_history)):
        version_num = len(article_history) - idx
        is_latest = (idx == 0)

        # Extract version type from article header
        if "Post-Debate" in article:
            version_type = "Post-Debate"
        elif "Post-Critique" in article:
            version_type = "Post-Critique"
        elif "User Feedback" in article:
            version_type = "User Feedback"
        elif "RESTORED ARTICLE (from v" in article:
            import re
            match = re.search(r'from v(\d+)', article)
            version_type = f"Restored v{match.group(1)}" if match else "Restored"
        elif "RESTORED ARTICLE" in article:
            version_type = "Restored"
        else:
            version_type = "Original"

        latest_class = " latest" if is_latest else ""
        selected_class = " selected" if is_latest else ""

        # Extract just the article content (remove headers)
        article_content = article
        if "---\n\n" in article:
            article_content = article.split("---\n\n", 1)[-1]
        elif "=" * 20 in article:
            # Handle initial article format (📝 YOUR ARTICLE\n====...\n\n)
            lines = article.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("=" * 20):
                    article_content = "\n".join(lines[i+1:]).lstrip()
                    break

        articles_data.append({
            "version": version_num,
            "type": version_type,
            "content": article_content
        })

        list_html += f"""
        <div class='version-item{latest_class}{selected_class}' data-version='{version_num}' onclick='selectVersion({version_num})'>
            <div class='version-label'>v{version_num} {"(Latest)" if is_latest else ""}</div>
            <div class='version-type'>{version_type}</div>
        </div>
        """

    # Get latest article content for initial preview
    latest_content = articles_data[0]["content"] if articles_data else ""

    return {
        "list": list_html,
        "articles": articles_data,
        "latest_content": latest_content
    }


def update_version_history(*args):
    """Wrapper to update version history, ignoring any input args from previous step."""
    import json
    return json.dumps(build_version_history_html())


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


def run_self_critique():
    """Run self-critique with chain-of-thought analysis streaming.

    Layout (matches Multi-Agent Debate):
    - Left panel: Critique analysis
    - Center panel: Original article → Refined article (transitions)
    - Right panel: Edit log (shows what changed)
    """
    global current_article_clean, current_grounding_context, article_history, original_article

    if not current_article_clean:
        error_msg = "⚠️ No article to critique. Please generate an article first."
        return error_msg, "", "", None

    # Store original for comparison later
    original_for_diff = current_article_clean

    # Center panel: original article (will transition to refined at end)
    center_display = "📄 ORIGINAL ARTICLE\n"
    center_display += "=" * 44 + "\n\n"
    center_display += original_article

    # Left panel: critique analysis (streaming)
    critique_analysis = "💭 SELF-CRITIQUE ANALYSIS\n"
    critique_analysis += "=" * 44 + "\n\n"
    critique_analysis += "🔎 Analyzing article for epistemic issues...\n\n"

    # Right panel: edit log placeholder
    edit_log_placeholder = "📝 EDIT LOG\n"
    edit_log_placeholder += "=" * 44 + "\n\n"
    edit_log_placeholder += "Changes will appear here after critique completes.\n\n"
    edit_log_placeholder += "The refined article will be compared to the original, "
    edit_log_placeholder += "and this panel will show exactly what was changed."

    yield critique_analysis, center_display, edit_log_placeholder, None

    try:
        # Step 1: Generate critique analysis with streaming
        critique_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        critique_chat.append(system(
            "You are a professional epistemic analyst conducting a critical evaluation of the article. "
            "Provide a structured analysis identifying weaknesses in epistemic quality.\n\n"
            "CONSTRAINTS:\n"
            "- Maximum 300 words total\n"
            "- Identify 2-3 most critical issues\n"
            "- Use formal, professional language\n"
            "- Be direct and specific in identifying problems\n\n"
            "Format your analysis as numbered sections (1, 2, 3...) addressing the most critical issues found. "
            "Consider evaluating:\n"
            "- Source reliability and diversification\n"
            "- Claims requiring qualification or additional context\n"
            "- Potential biases or one-sided framing\n"
            "- Missing caveats or epistemic humility\n"
            "- Overstatements or unwarranted certainty\n\n"
            "Use clear, declarative statements rather than casual observations. "
            "Number your sections sequentially (1, 2, 3) based on the issues you identify."
        ))
        critique_chat.append(user(
            f"Analyze this article for epistemic issues:\n\n{current_article_clean}"
        ))

        # Stream the critique analysis
        full_critique = ""
        for response, chunk in critique_chat.stream():
            if chunk.content:
                full_critique += chunk.content
                critique_analysis_display = "💭 SELF-CRITIQUE ANALYSIS\n"
                critique_analysis_display += "=" * 44 + "\n\n"
                critique_analysis_display += full_critique
                yield critique_analysis_display, center_display, edit_log_placeholder, None

        # Step 2: Generate refined article based on critique
        critique_analysis_display = "💭 SELF-CRITIQUE ANALYSIS\n"
        critique_analysis_display += "=" * 44 + "\n\n"
        critique_analysis_display += full_critique + "\n\n"
        critique_analysis_display += "✓ Analysis complete. Generating refined article...\n"
        yield critique_analysis_display, center_display, edit_log_placeholder, None

        refinement_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        refinement_chat.append(system(
            "You are refining an article based on epistemic critique. Your task:\n"
            "- Address all valid concerns raised in the critique\n"
            "- Add appropriate qualifiers and nuance\n"
            "- Strengthen unsupported claims with evidence or soften them\n"
            "- Maintain factual accuracy and cite sources\n"
            "- Keep the same structure and ~300 word length\n"
            "Output ONLY the final refined article in markdown format with inline citations and Sources section."
        ))
        refinement_chat.append(user(
            f"Original article:\n{current_article_clean}\n\n"
            f"Critique analysis:\n{full_critique}\n\n"
            f"Produce the refined article addressing these concerns."
        ))

        # Show generating status in edit log
        generating_edit_log = "📝 EDIT LOG\n"
        generating_edit_log += "=" * 44 + "\n\n"
        generating_edit_log += "Generating refined article...\n\n"
        generating_edit_log += "Edit log will be computed once refinement is complete."

        # Stream refined article to CENTER panel (replacing original)
        refined_header = "📝 REFINED ARTICLE\n" + "=" * 44 + "\n\n"
        refined_article = ""
        for response, chunk in refinement_chat.stream():
            if chunk.content:
                refined_article += chunk.content
                streaming_center = refined_header + refined_article
                yield critique_analysis_display, streaming_center, generating_edit_log, None

        # Generate the edit log by comparing original and revised
        edit_log = generate_edit_log(original_for_diff, refined_article)

        # Update current article for iterative critiques
        current_article_clean = refined_article

        # Store refined article in history
        final_article = f"📝 REFINED ARTICLE (Post-Critique)\n"
        final_article += "=" * 44 + "\n\n"
        final_article += f"__Epistemically refined through self-critique__\n\n---\n\n{refined_article}"
        article_history.append(final_article)

        # Save refined article to file for download
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"critique_{timestamp}.md"
        filepath = os.path.abspath(filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(refined_article)
        except Exception as e:
            print(f"Error saving refined article: {e}")
            filepath = None

        # Final yield with complete versions
        final_critique_display = "💭 SELF-CRITIQUE ANALYSIS\n"
        final_critique_display += "=" * 44 + "\n\n"
        final_critique_display += full_critique + "\n\n"
        final_critique_display += "✅ Critique complete! Refined article generated."

        final_center = refined_header + refined_article
        yield final_critique_display, final_center, edit_log, filepath

    except Exception as e:
        error_analysis = critique_analysis + \
            f"\n\n❌ Error during critique: {str(e)}\n\nPlease try again."
        yield error_analysis, center_display, edit_log_placeholder, None


def run_user_feedback(user_feedback):
    """Process user feedback on current article with validation and critical evaluation."""
    global current_article_clean, current_grounding_context, article_history, original_article

    if not current_article_clean:
        error_msg = "⚠️ No article available. Please generate an article first."
        center_display = "📝 CURRENT ARTICLE\n" + "=" * 44 + "\n\n" + \
            "No article available. Generate an article before providing feedback."
        rejection_display = "❌ FEEDBACK REJECTED\n" + "=" * 44 + "\n\n" + \
            "No article available to provide feedback on.\n\nGenerate an article first using Article Generation."
        return center_display, error_msg, rejection_display, None

    # Store original article for comparison
    original_article_text = current_article_clean

    # Center panel: current article (read-only reference)
    center_display = "📝 CURRENT ARTICLE\n"
    center_display += "=" * 44 + "\n\n"
    center_display += current_article_clean

    # Left panel: show processing message temporarily
    processing_msg = "⏳ Processing your feedback...\n\nValidating suggestions against source material..."

    # Right panel: placeholder during processing
    right_placeholder = "📋 CHANGELOG\n" + "=" * 44 + \
        "\n\n⏳ Processing...\n\nChangelog will appear here."

    yield center_display, processing_msg, right_placeholder, None

    try:
        # Step 1: Validate and analyze user feedback
        validation_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        validation_chat.append(system(
            "You are a feedback validation agent. Your task is to analyze user feedback about an article "
            "and determine if it contains actionable, substantive suggestions.\n\n"
            "Feedback is considered VALID if it:\n"
            "- Relates to the article's content, structure, or epistemic quality\n"
            "- Provides specific suggestions or identifies specific issues\n"
            "- Is coherent and understandable\n\n"
            "Feedback is considered INVALID if it:\n"
            "- Is empty, too short (< 10 characters), or pure gibberish\n"
            "- Is completely unrelated to the article\n"
            "- Contains no actionable suggestions\n\n"
            "If VALID: Identify and number each distinct suggestion (1, 2, 3, etc.)\n"
            "If INVALID: Explain why and provide guidance on what constitutes good feedback.\n\n"
            "Format your response as:\n"
            "VALIDATION: [VALID or INVALID]\n"
            "REASON: [Brief explanation]\n"
            "SUGGESTIONS: [If valid, numbered list of extracted suggestions]"
        ))
        validation_chat.append(user(
            f"Article:\n{current_article_clean}\n\n"
            f"User feedback:\n{user_feedback}\n\n"
            f"Validate this feedback and extract actionable suggestions."
        ))

        # Get validation result
        validation_response = ""
        for response, chunk in validation_chat.stream():
            if chunk.content:
                validation_response += chunk.content

        # Check if feedback is valid
        is_valid = "VALIDATION: VALID" in validation_response

        if not is_valid:
            # Feedback rejected - keep user's feedback in left panel (editable), put rejection in right panel
            rejection_display = "❌ FEEDBACK REJECTED\n"
            rejection_display += "=" * 44 + "\n\n"
            rejection_display += f"{validation_response}\n\n"
            rejection_display += "---\n\n"
            rejection_display += "💡 Tips for good feedback:\n\n"
            rejection_display += "- Reference specific claims or sections\n"
            rejection_display += "- Suggest concrete improvements\n"
            rejection_display += "- Focus on epistemic quality (certainty language, sources, framing)\n"
            rejection_display += "- Be clear and specific\n\n"
            rejection_display += "Edit your feedback in the left panel and try again."

            # Keep user's original feedback in left panel (still editable)
            yield center_display, user_feedback, rejection_display, None
            return

        # Step 2: Feedback is valid - evaluate suggestions against sources
        evaluation_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        evaluation_chat.append(system(
            "You are an epistemic evaluation agent. Analyze user feedback suggestions and determine which ones "
            "should be incorporated into the article revision.\n\n"
            "For each suggestion:\n"
            "- Evaluate it against the source material and epistemic standards\n"
            "- Mark it as: ✓ (incorporate), ⚠️ (partially valid - modify), or ✗ (reject)\n"
            "- Provide brief reasoning\n\n"
            "Use this format:\n"
            "SUGGESTION 1: [User's suggestion]\n"
            "DECISION: [✓/⚠️/✗]\n"
            "REASONING: [Why this decision was made]\n\n"
            "Be critical but fair. Accept valid improvements, push back on unsupported claims, "
            "and modify suggestions to align with epistemic integrity."
        ))
        evaluation_chat.append(user(
            f"Article:\n{current_article_clean}\n\n"
            f"Source context:\n{current_grounding_context}\n\n"
            f"User feedback:\n{user_feedback}\n\n"
            f"Validation analysis:\n{validation_response}\n\n"
            f"Evaluate each suggestion and determine which should be incorporated."
        ))

        # Get evaluation result (non-streaming for changelog generation)
        evaluation_result = ""
        for response, chunk in evaluation_chat.stream():
            if chunk.content:
                evaluation_result += chunk.content

        # Step 3: Generate revised article incorporating valid feedback
        revision_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        revision_chat.append(system(
            "You are revising an article based on evaluated user feedback. "
            "Incorporate suggestions marked with ✓ (fully), suggestions marked with ⚠️ (with modifications), "
            "and ignore suggestions marked with ✗.\n\n"
            "Maintain:\n"
            "- Factual accuracy and source alignment\n"
            "- Epistemic integrity\n"
            "- Same structure and ~300 word length\n"
            "- Inline citations and Sources section\n\n"
            "Output ONLY the final revised article in markdown format."
        ))
        revision_chat.append(user(
            f"Original article:\n{original_article_text}\n\n"
            f"Feedback evaluation:\n{evaluation_result}\n\n"
            f"Source context:\n{current_grounding_context}\n\n"
            f"Produce the revised article incorporating the approved suggestions."
        ))

        # Stream revised article to CENTER panel
        revised_article = ""
        revision_header = "📝 CURRENT ARTICLE\n" + "=" * 44 + "\n\n"

        for response, chunk in revision_chat.stream():
            if chunk.content:
                revised_article += chunk.content
                streaming_center = revision_header + revised_article
                yield streaming_center, processing_msg, right_placeholder, None

        # Step 4: Generate changelog showing what changed
        changelog_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        changelog_chat.append(system(
            "You are generating a changelog that shows what changed in the article based on user feedback.\n\n"
            "Create a clear, structured changelog with these sections:\n"
            "✅ ACCEPTED - Suggestions that were fully incorporated\n"
            "⚠️ PARTIALLY ACCEPTED - Suggestions that were modified before incorporating\n"
            "❌ REJECTED - Suggestions that were not incorporated\n\n"
            "For each item, include:\n"
            "- The user's original feedback/suggestion\n"
            "- What actually changed in the article (be specific)\n"
            "- Brief reasoning for the decision\n\n"
            "Keep it concise but informative. Use clear formatting with bullet points or numbered lists."
        ))
        changelog_chat.append(user(
            f"Original article:\n{original_article_text}\n\n"
            f"Revised article:\n{revised_article}\n\n"
            f"User feedback:\n{user_feedback}\n\n"
            f"Evaluation:\n{evaluation_result}\n\n"
            f"Generate a changelog showing what changed and why."
        ))

        # Stream changelog to RIGHT panel
        changelog = ""
        changelog_header = "📋 CHANGELOG\n" + "=" * 44 + "\n\n"

        for response, chunk in changelog_chat.stream():
            if chunk.content:
                changelog += chunk.content
                streaming_changelog = changelog_header + changelog
                final_center = revision_header + revised_article
                yield final_center, processing_msg, streaming_changelog, None

        # Update current article
        current_article_clean = revised_article

        # Store in history
        final_article = f"📝 REVISED ARTICLE (User Feedback)\n"
        final_article += "=" * 44 + "\n\n"
        final_article += f"__Revised based on user feedback__\n\n---\n\n{revised_article}"
        article_history.append(final_article)

        # Save revised article to file for download
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_{timestamp}.md"
        filepath = os.path.abspath(filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(revised_article)
        except Exception as e:
            print(f"Error saving feedback revised article: {e}")
            filepath = None

        # Final display - use placeholder for "ready for next round" message
        final_center = revision_header + revised_article
        final_changelog = changelog_header + changelog

        # Return with empty value and placeholder for next round
        yield final_center, gr.update(value="", placeholder="✅ Feedback processed! Article updated.\n\nEnter new feedback to continue refining..."), final_changelog, filepath

    except Exception as e:
        error_display = f"❌ Error processing feedback: {str(e)}\n\nPlease try again."
        yield center_display, user_feedback, error_display, None


def run_synthetic_data_generation(topic, num_examples, quality_dist, flaw_type, article_length):
    """Generate synthetic training data with controlled epistemic properties."""
    import json
    import datetime

    # Validate topic is not empty
    if not topic or topic.strip() == "":
        error_msg = "⚠️ ERROR: Please enter a topic before generating synthetic data."
        yield error_msg, "", "", None
        return

    # Comprehensive topic validation
    import re
    topic_clean = topic.strip()

    # Check minimum length
    if len(topic_clean) < 2:
        error_msg = "⚠️ ERROR: Topic must be at least 2 characters long.\n\nPlease enter a meaningful topic (e.g., 'AI', 'Climate Change', 'Quantum Computing')."
        yield error_msg, "", "", None
        return

    # Check for alphabetic content (reject pure emojis, punctuation, numbers)
    alphabetic_chars = re.findall(r'[a-zA-Z]', topic_clean)
    if len(alphabetic_chars) < 2:
        error_msg = "⚠️ ERROR: Topic must contain at least 2 letters.\n\nPlease enter a meaningful topic (e.g., 'AI', 'Climate Change', 'Quantum Computing')."
        yield error_msg, "", "", None
        return

    # Check for variety (reject single repeated character like "aaaa" or "kkkk")
    if len(set(alphabetic_chars)) < 2:
        error_msg = "⚠️ ERROR: Topic must contain more than one unique letter.\n\nPlease enter a meaningful topic (e.g., 'AI', 'Climate Change', 'Quantum Computing')."
        yield error_msg, "", "", None
        return

    # Convert num_examples from string to int
    num_examples = int(num_examples)

    # Map article length to word count range
    if article_length.startswith("Brief"):
        word_range = "75-100"
    elif article_length.startswith("Standard"):
        word_range = "175-200"
    elif article_length.startswith("Long"):
        word_range = "275-300"
    else:
        word_range = "175-200"  # Default to Standard

    # Convert quality distribution selection to quality tiers list
    if quality_dist == "All":
        quality_tiers = ["excellent", "good", "fair", "poor", "terrible"]
    elif quality_dist == "Excellent":
        quality_tiers = ["excellent"]
    elif quality_dist == "Good":
        quality_tiers = ["good"]
    elif quality_dist == "Poor":
        quality_tiers = ["poor"]
    elif quality_dist == "Terrible":
        quality_tiers = ["terrible"]
    else:
        quality_tiers = ["excellent", "good", "fair", "poor", "terrible"]

    # Left panel: Generation log
    log = "🏭 SYNTHETIC DATA GENERATION\n"
    log += "=" * 44 + "\n\n"
    log += f"📊 Configuration:\n"
    log += f"- Target topic: {topic}\n"
    log += f"- Number of examples: {num_examples}\n"
    log += f"- Quality distribution: {quality_dist}\n"
    log += f"- Flaw type: {flaw_type}\n"
    log += f"- Article length: {article_length} ({word_range} words)\n\n"
    log += "⏳ Starting generation process...\n\n"

    yield "", log, "", None

    # Storage for generated data
    synthetic_dataset = []

    # Center panel: Article preview (will update with each generation)
    center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"

    # Right panel: Metadata accumulation
    metadata_display = "📋 DATASET METADATA\n" + "=" * 44 + "\n\n"

    try:
        for i in range(num_examples):
            # Select quality tier and specific flaw for this example
            quality = quality_tiers[i % len(quality_tiers)]

            # Determine target flaw based on flaw_type selection
            # Excellent quality always has no flaws, regardless of flaw_type
            if quality == "excellent":
                target_flaw = "none"
            elif flaw_type == "Auto":
                # Map quality to specific epistemic flaws automatically
                flaw_map = {
                    "good": "minor_overcertainty",
                    "fair": "missing_citations",
                    "poor": "multiple_flaws",
                    "terrible": "severe_bias_and_overstatements"
                }
                target_flaw = flaw_map.get(quality, "none")
            elif flaw_type == "Citations":
                target_flaw = "missing_citations"
            elif flaw_type == "Certainty":
                target_flaw = "overcertainty"
            elif flaw_type == "Bias":
                target_flaw = "biased_framing"
            elif flaw_type == "Multiple":
                target_flaw = "multiple_flaws"
            else:
                target_flaw = "auto"

            # Update log
            log += f"📝 Generating example {i+1}/{num_examples}...\n"
            log += f"   Quality tier: {quality}\n"
            log += f"   Target flaw: {target_flaw}\n\n"
            yield center_preview, log, metadata_display, None

            # Generate article with controlled epistemic quality
            generator_chat = client.chat.create(
                model="grok-4-1-fast-reasoning")

            # Craft prompt based on target quality and specific flaw type
            if quality == "excellent":
                # Excellent articles are always truly excellent, ignore flaw_type
                quality_instruction = """Generate an epistemically EXCELLENT article (should score 9-10 on all dimensions):
- Use precise hedging language ("suggests", "indicates", "may", "appears to") for uncertain claims
- Include specific citations with sources for ALL major claims (e.g., "According to [Source, Year]...")
- Present multiple perspectives on any debatable points
- Explicitly distinguish between established facts and ongoing research/speculation
- Include caveats about data limitations or methodological constraints
- Use measured, qualified language throughout - no absolute statements without ironclad evidence
- Include a Sources/References section with specific attributions"""
            elif quality == "good":
                quality_instruction = """Generate a GOOD quality article (target 6-8 scores):
- Include SPECIFIC citations for most claims (e.g., "According to [Source, Year]...")
- Use appropriate hedging language for most statements ("suggests", "indicates", "research shows")
- Allow only 1-2 minor lapses: one claim without citation OR one slightly promotional phrase ("vibrant", "boasts")
- Maintain neutral, informative tone overall - avoid multiple promotional words
- Present facts objectively without one-sided framing
- Include a proper Sources section with multiple references
- The article should feel solid and informative, just not quite perfect"""
            elif quality == "fair":
                quality_instruction = """Generate a FAIR quality article (target 4-6 scores):
- Include SOME specific citations, but leave several significant claims unsourced
- Mix hedging with definitive statements - use "may" and "suggests" sometimes, but also use "is" and "proves" for uncertain claims
- Attempt balance but lean slightly positive/promotional without being one-sided propaganda
- Include basic sourcing attempts ("studies show", "researchers say") even if vague
- Include a Sources section but it can be incomplete or vague
- Should feel like a decent Wikipedia article with noticeable gaps"""
            elif quality == "poor":
                quality_instruction = """Generate an article with significant epistemic problems:
- Many claims lack sources or citations
- Overconfident language throughout ("X is", "Y proves", etc.)
- Missing important qualifiers and hedges
- One-sided framing on debatable topics
- Presents speculation as fact in multiple places"""
            else:  # terrible
                quality_instruction = """Generate an article with severe epistemic failures:
- Almost no citations or sources provided
- Extreme overconfidence and absolutist language
- No hedging or uncertainty acknowledgment
- Heavily biased framing
- Speculation presented as definitive fact
- Makes strong causal claims without evidence"""

            # Add flaw-specific emphasis only for non-excellent quality tiers
            if quality != "excellent" and flaw_type != "Auto":
                if flaw_type == "Citations":
                    quality_instruction += "\n\nIMPORTANT: Focus especially on citation/sourcing issues. Make claims without proper attribution or sources."
                elif flaw_type == "Certainty":
                    quality_instruction += "\n\nIMPORTANT: Focus especially on certainty language issues. Use overconfident language and avoid appropriate hedging."
                elif flaw_type == "Bias":
                    quality_instruction += "\n\nIMPORTANT: Focus especially on biased framing. Present one-sided perspectives and miss important counterarguments."
                elif flaw_type == "Multiple":
                    quality_instruction += "\n\nIMPORTANT: Include multiple types of epistemic flaws (citations, certainty, bias) throughout the article."

            generator_chat.append(system(
                f"You are generating a synthetic training example for an epistemic quality classifier.\n\n"
                f"{quality_instruction}\n\n"
                f"Write a {word_range} word article about the given topic with the specified epistemic characteristics. "
                f"Output ONLY the article text in markdown format. Include a brief Sources section if appropriate."
            ))
            generator_chat.append(user(f"Generate an article about: {topic}"))

            # Generate article with streaming updates
            generated_article = ""
            for response, chunk in generator_chat.stream():
                if chunk.content:
                    generated_article += chunk.content

                    # Rebuild center preview with streaming article
                    temp_center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"
                    # Show all previously completed articles
                    for entry in synthetic_dataset:
                        temp_center_preview += f"**Example {entry['id']}/{num_examples}** (Quality: {entry['target_quality'].upper()})\n"
                        temp_center_preview += "-" * 44 + "\n\n"
                        temp_center_preview += entry['article'] + "\n\n"
                        temp_center_preview += "=" * 44 + "\n\n"
                    # Show currently streaming article
                    temp_center_preview += f"**Example {i+1}/{num_examples}** (Quality: {quality.upper()}) - GENERATING...\n"
                    temp_center_preview += "-" * 44 + "\n\n"
                    temp_center_preview += generated_article + \
                        "▌"  # Add cursor to show it's streaming

                    yield temp_center_preview, log, metadata_display, None

            # Label the article with epistemic scores
            labeler_chat = client.chat.create(model="grok-4-1-fast-reasoning")
            labeler_chat.append(system(
                "You are an epistemic quality labeler. Analyze the article and provide scores (0-10) for:\n"
                "- source_quality: How well claims are sourced and cited\n"
                "- certainty_appropriateness: Whether certainty language matches evidence strength\n"
                "- bias_level: How balanced vs biased the framing is (10 = very balanced, 0 = very biased)\n"
                "- completeness: Whether important caveats and limitations are mentioned\n\n"
                "Evaluate objectively using the full 0-10 scale:\n"
                "- 9-10: Near-perfect epistemic quality\n"
                "- 7-8: Strong quality with minor issues\n"
                "- 5-6: Adequate with noticeable problems\n"
                "- 3-4: Significant issues but some structure\n"
                "- 1-2: Severe problems, minimal value\n"
                "- 0: Complete epistemic failure\n\n"
                "Also identify specific epistemic flaws present.\n\n"
                "Format your response as:\n"
                "SOURCE_QUALITY: [0-10]\n"
                "CERTAINTY: [0-10]\n"
                "BIAS: [0-10]\n"
                "COMPLETENESS: [0-10]\n"
                "FLAWS: [comma-separated list of specific issues]"
            ))
            labeler_chat.append(
                user(f"Label this article:\n\n{generated_article}"))

            # Get labels
            label_response = ""
            for response, chunk in labeler_chat.stream():
                if chunk.content:
                    label_response += chunk.content

            # Parse labels
            import re
            source_quality = int(re.search(r'SOURCE_QUALITY:\s*(\d+)', label_response).group(
                1)) if re.search(r'SOURCE_QUALITY:\s*(\d+)', label_response) else 5
            certainty = int(re.search(r'CERTAINTY:\s*(\d+)', label_response).group(1)
                            ) if re.search(r'CERTAINTY:\s*(\d+)', label_response) else 5
            bias = int(re.search(r'BIAS:\s*(\d+)', label_response).group(1)
                       ) if re.search(r'BIAS:\s*(\d+)', label_response) else 5
            completeness = int(re.search(r'COMPLETENESS:\s*(\d+)', label_response).group(
                1)) if re.search(r'COMPLETENESS:\s*(\d+)', label_response) else 5
            flaws_match = re.search(r'FLAWS:\s*(.+)', label_response)
            flaws = flaws_match.group(1).strip(
            ) if flaws_match else "none identified"

            # Store in dataset
            data_entry = {
                "id": i + 1,
                "topic": topic,
                "article": generated_article,
                "target_quality": quality,
                "target_flaw": target_flaw,
                "labels": {
                    "source_quality": source_quality,
                    "certainty_appropriateness": certainty,
                    "bias_balance": bias,
                    "completeness": completeness
                },
                "identified_flaws": flaws,
                "generated_at": datetime.datetime.now().isoformat()
            }
            synthetic_dataset.append(data_entry)

            # Update center preview with all articles
            center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"
            for entry in synthetic_dataset:
                center_preview += f"**Example {entry['id']}/{num_examples}** (Quality: {entry['target_quality'].upper()})\n"
                center_preview += "-" * 44 + "\n\n"
                center_preview += entry['article'] + "\n\n"
                center_preview += "=" * 44 + "\n\n"

            # Update metadata display
            metadata_display = "📋 DATASET METADATA\n" + "=" * 44 + "\n\n"
            for entry in synthetic_dataset:
                metadata_display += f"**Example {entry['id']}** - {entry['target_quality'].upper()}\n"
                metadata_display += f"Source Quality: {entry['labels']['source_quality']}/10\n"
                metadata_display += f"Certainty Appropriateness: {entry['labels']['certainty_appropriateness']}/10\n"
                metadata_display += f"Bias Balance: {entry['labels']['bias_balance']}/10\n"
                metadata_display += f"Completeness: {entry['labels']['completeness']}/10\n"
                metadata_display += f"Flaws: {entry['identified_flaws']}\n\n"

            # Update log
            log += f"✅ Example {i+1} complete\n"
            log += f"   Scores: {source_quality}, {certainty}, {bias}, {completeness}\n\n"

            yield center_preview, log, metadata_display, None

        # Export to JSONL
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize topic for filename (remove invalid characters)
        import re
        import os
        safe_topic = re.sub(r'[<>:"/\\|?*]', '', topic)  # Remove invalid chars
        # Replace spaces with underscores
        safe_topic = safe_topic.replace(' ', '_')
        # Limit length to avoid overly long filenames
        safe_topic = safe_topic[:50]
        filename = f"synthetic_data_{safe_topic}_{timestamp}.jsonl"

        # Convert to absolute path for reliable downloads in deployed environments
        absolute_filepath = os.path.abspath(filename)

        with open(absolute_filepath, 'w', encoding='utf-8') as f:
            for entry in synthetic_dataset:
                f.write(json.dumps(entry) + '\n')

        # Final log update
        log += "\n" + "=" * 44 + "\n"
        log += f"✅ Generation complete!\n\n"
        log += f"📁 Exported to: {filename}\n"
        log += f"📊 Total examples: {num_examples}\n"
        log += f"💾 Format: JSONL (for ML training)\n\n"
        log += "Dataset can be used for:\n"
        log += "- Training epistemic classifiers\n"
        log += "- Evaluating quality detection systems\n"
        log += "- Fine-tuning article generators\n"

        # Final metadata summary
        metadata_display += "\n" + "=" * 44 + "\n"
        metadata_display += f"**📊 DATASET SUMMARY**\n\n"
        metadata_display += f"Total examples: {num_examples}\n"
        metadata_display += f"Topic: {topic}\n"
        metadata_display += f"Quality distribution:\n"
        for tier in quality_tiers:
            count = sum(
                1 for e in synthetic_dataset if e['target_quality'] == tier)
            metadata_display += f"  - {tier}: {count}\n"
        metadata_display += f"\n📁 Saved to: {filename}\n"

        yield center_preview, log, metadata_display, absolute_filepath

    except Exception as e:
        error_log = log + \
            f"\n\n❌ Error during generation: {str(e)}\n\nPlease try again."
        yield center_preview, error_log, metadata_display, None


def generate_edit_log(original: str, revised: str) -> str:
    """Generate a structured edit log comparing original and revised text.

    Uses section-aware sentence comparison to show what changed.
    """
    import re

    def truncate(text, max_len=140):
        """Truncate text to max_len with ellipsis."""
        text = text.replace('\n', ' ').strip()
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."

    def count_sources(text):
        """Count the number of sources in the text."""
        source_patterns = re.findall(r'\[[\^]?\d+\]:', text)
        if source_patterns:
            return len(source_patterns)
        if "## Sources" in text:
            sources_section = text.split("## Sources")[-1]
            lines = [l.strip() for l in sources_section.strip().split(
                '\n') if l.strip() and l.strip() != '## Sources']
            return len(lines)
        return 0

    def split_into_sections(text):
        """Split text into sections by ## headers."""
        sections = {}
        current_header = "intro"
        current_content = []

        for line in text.split('\n'):
            if line.startswith('## '):
                if current_content:
                    sections[current_header] = '\n'.join(current_content)
                current_header = line.strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_header] = '\n'.join(current_content)

        return sections

    def split_sentences(text):
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    # Count sources for summary
    original_source_count = count_sources(original)
    revised_source_count = count_sources(revised)

    # Split into sections
    orig_sections = split_into_sections(original)
    rev_sections = split_into_sections(revised)

    # Compare sentences within each section (preserve order from original)
    changes = []
    all_headers = list(orig_sections.keys())
    for h in rev_sections.keys():
        if h not in all_headers:
            all_headers.append(h)

    for header in all_headers:
        # Skip sources section
        if 'Sources' in header or 'sources' in header:
            continue

        orig_text = orig_sections.get(header, "")
        rev_text = rev_sections.get(header, "")

        orig_sentences = split_sentences(orig_text)
        rev_sentences = split_sentences(rev_text)

        # Compare by position within this section
        max_len = max(len(orig_sentences), len(rev_sentences))
        for i in range(max_len):
            orig_sent = orig_sentences[i] if i < len(orig_sentences) else ""
            rev_sent = rev_sentences[i] if i < len(rev_sentences) else ""

            # Skip if identical or both empty
            if orig_sent == rev_sent:
                continue
            # Skip if one is empty
            if not orig_sent or not rev_sent:
                continue

            changes.append((orig_sent, rev_sent))

    # Build edit log
    edit_log = "📝 EDIT LOG\n"
    edit_log += "=" * 44 + "\n\n"

    if changes:
        for orig, rev in changes:
            edit_log += "                 (Original)\n"
            edit_log += f"{truncate(orig)}\n\n"
            edit_log += "                    ↓\n\n"
            edit_log += "                 (Revised)\n"
            edit_log += f"{truncate(rev)}\n\n"
            edit_log += "-" * 44 + "\n\n"

    # Summary
    if changes:
        edit_log += f"Total: {len(changes)} edits\n"
    else:
        edit_log += "No significant changes detected.\n"

    # Note source changes if there are any
    if revised_source_count != original_source_count:
        edit_log += f"Sources expanded from {original_source_count} to {revised_source_count}\n"

    return edit_log


def run_multi_agent_debate():
    """Run multi-agent debate on current article.

    Layout:
    - Left panel: Debate transcript
    - Center panel: Original article → Revised article (transitions)
    - Right panel: Edit log (shows what changed)
    """
    global current_article_clean, current_grounding_context, article_history, original_article

    if not current_article_clean:
        error_msg = "⚠️ No article to debate. Please generate an article first."
        return error_msg, "", "", None

    # Store original for comparison later
    original_for_diff = current_article_clean

    # Center panel: original article (will transition to revised at end)
    center_display = "📄 ORIGINAL ARTICLE\n"
    center_display += "=" * 44 + "\n\n"
    center_display += original_article

    # Left panel: debate transcript (progressive updates)
    debate_transcript = "🎭 DEBATE TRANSCRIPT\n"
    debate_transcript += "=" * 44 + "\n\n"
    debate_transcript += "⏳ Initializing debate agents...\n\n"

    # Right panel: edit log placeholder
    edit_log_placeholder = "📝 EDIT LOG\n"
    edit_log_placeholder += "=" * 44 + "\n\n"
    edit_log_placeholder += "Changes will appear here after the debate concludes.\n\n"
    edit_log_placeholder += "The arbiter will synthesize the debate into a revised article, "
    edit_log_placeholder += "and this panel will show exactly what was changed."

    yield debate_transcript, center_display, edit_log_placeholder, None

    # Agent 1: Defender
    debate_transcript = "🎭 DEBATE TRANSCRIPT\n"
    debate_transcript += "=" * 44 + "\n\n"
    debate_transcript += "🟢 DEFENDER AGENT\n"
    debate_transcript += "-" * 44 + "\n"
    debate_transcript += "⏳ Analyzing article for strengths...\n\n"
    yield debate_transcript, center_display, edit_log_placeholder, None

    try:
        defender_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        defender_chat.append(system(
            "You are the Defender agent in an epistemic debate. Your role is to identify the strongest "
            "epistemic qualities of the article: appropriate certainty language, balanced framing, "
            "acknowledgment of limitations, and clear communication. Argue for what the article does well "
            "in terms of epistemic integrity. Be rigorous but fair. Provide a concise defense (~150 words)."
        ))
        defender_chat.append(user(
            f"Defend the epistemic strengths of this article:\n\n{current_article_clean}"))

        # Stream Defender response
        defender_response = ""
        debate_base = "🎭 DEBATE TRANSCRIPT\n" + "=" * \
            44 + "\n\n🟢 DEFENDER AGENT\n" + "-" * 44 + "\n"

        for response, chunk in defender_chat.stream():
            if chunk.content:
                defender_response += chunk.content
                streaming_transcript = debate_base + defender_response
                yield streaming_transcript, center_display, edit_log_placeholder, None

        debate_transcript = debate_base + f"{defender_response}\n\n\n"
        yield debate_transcript, center_display, edit_log_placeholder, None

        # Agent 2: Challenger
        debate_transcript += "🔴 CHALLENGER AGENT\n"
        debate_transcript += "-" * 44 + "\n"
        debate_transcript += "⏳ Analyzing article for weaknesses...\n\n"
        yield debate_transcript, center_display, edit_log_placeholder, None

        challenger_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        challenger_chat.append(system(
            "You are the Challenger agent in an epistemic debate. Your role is to critically examine "
            "the article for epistemic weaknesses: overstatements, unwarranted certainty, missing caveats, "
            "one-sided framing, lack of epistemic humility, or unclear communication. Focus on how claims "
            "are presented, not their factual accuracy. Be aggressive but fair. Point out specific issues. "
            "Provide a concise critique (~150 words)."
        ))
        challenger_chat.append(user(
            f"Challenge the epistemic quality of this article:\n\n{current_article_clean}"))

        # Stream Challenger response
        challenger_response = ""
        challenger_base = debate_transcript[:-
                                            len("⏳ Analyzing article for weaknesses...\n\n")]

        for response, chunk in challenger_chat.stream():
            if chunk.content:
                challenger_response += chunk.content
                streaming_transcript = challenger_base + challenger_response
                yield streaming_transcript, center_display, edit_log_placeholder, None

        debate_transcript = challenger_base + f"{challenger_response}\n\n\n"
        yield debate_transcript, center_display, edit_log_placeholder, None

        # Agent 3: Arbiter (produces revised article)
        debate_transcript += "⚖️ ARBITER AGENT\n"
        debate_transcript += "-" * 44 + "\n"
        debate_transcript += "⏳ Synthesizing debate and producing revised article...\n\n"
        yield debate_transcript, center_display, edit_log_placeholder, None

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

Produce the final revised article."""))

        # Stream Arbiter's revised article to CENTER panel (replacing original)
        revised_article = ""
        revised_header = "📝 REVISED ARTICLE\n" + "=" * 44 + "\n\n"

        # Show "generating" in edit log
        generating_edit_log = "📝 EDIT LOG\n"
        generating_edit_log += "=" * 44 + "\n\n"
        generating_edit_log += "Generating revised article...\n\n"
        generating_edit_log += "Edit log will be computed once revision is complete."

        for response, chunk in arbiter_chat.stream():
            if chunk.content:
                revised_article += chunk.content
                streaming_center = revised_header + revised_article
                yield debate_transcript, streaming_center, generating_edit_log, None

        # Generate the edit log by comparing original and revised
        edit_log = generate_edit_log(original_for_diff, revised_article)

        # Add completion message to debate transcript
        debate_transcript += "✅ Debate complete! Revised article generated.\n"

        # Center panel: final revised article
        final_center = revised_header + revised_article

        # Update current article to the revision for iterative debates
        current_article_clean = revised_article

        # Store revised article in history
        final_article = f"📝 REVISED ARTICLE (Post-Debate)\n"
        final_article += "=" * 44 + "\n\n"
        final_article += revised_article
        article_history.append(final_article)

        # Save revised article to file for download
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debate_{timestamp}.md"
        filepath = os.path.abspath(filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(revised_article)
        except Exception as e:
            print(f"Error saving revised article: {e}")
            filepath = None

        yield debate_transcript, final_center, edit_log, filepath

    except Exception as e:
        error_transcript = debate_transcript + \
            f"\n\n❌ Error during debate: {str(e)}\n\nPlease try again."
        yield error_transcript, center_display, edit_log_placeholder, None


def fetch_content_from_url(url: str) -> Optional[Dict[str, str]]:
    """Fetch content from a URL. Supports Wikipedia URLs via API."""
    try:
        if "wikipedia.org/wiki/" in url:
            from urllib.parse import unquote
            title = url.split("/wiki/")[-1].replace("_", " ")
            title = unquote(title)

            api_url = "https://en.wikipedia.org/w/api.php"
            headers = {
                "User-Agent": "Veritas-Epistemics/1.0 (Educational Project)"}
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True
            }
            response = requests.get(
                api_url, params=params, headers=headers, timeout=10)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            page_id = list(pages.keys())[0]

            if page_id == "-1":
                return {"title": title, "content": "", "url": url}

            extract = pages[page_id].get("extract", "")
            actual_title = pages[page_id].get("title", title)

            return {"title": actual_title, "content": extract, "url": url}
        else:
            return {"title": url, "content": "", "url": url}
    except Exception as e:
        print(f"Error fetching content from URL: {e}")
        return {"title": url, "content": "", "url": url}


def generate_initial_article(topic: str):
    """Generate initial article with Wikipedia-first optimization (web search fallback)."""
    global article_history, current_sources, current_article_clean, current_grounding_context, original_article

    # Clear version history for new topic
    article_history = []

    # Status in left panel
    status_log = "🔍 PROCESS LOG\n"
    status_log += "=" * 44 + "\n\n"
    status_log += "🚀 Starting article generation...\n\n"

    # Show loading message in center panel
    article_placeholder = "📝 YOUR ARTICLE\n" + "=" * 44 + \
        "\n\nYour generated article will appear shortly..."

    # Show loading message in right panel
    loading_message = "📚 SOURCE MATERIAL\n" + "=" * 44 + \
        "\n\nSearching for sources...\n\nSources will appear here when article generation is complete."

    yield article_placeholder, status_log, loading_message, None

    # Initialize variables
    source_notes = []
    current_sources = []
    source_context = ""
    streaming_sources_display = ""
    use_wikipedia = False
    wiki_data = None

    # Step 1: Try Wikipedia first (fast path)
    status_log += "🔎 Searching for sources...\n\n"
    yield article_placeholder, status_log, loading_message, None

    wiki_data = fetch_wikipedia(topic)  # This API call IS the natural delay

    if wiki_data:
        # Check match quality - calculate word overlap
        topic_words = set(topic.lower().split())
        title_words = set(wiki_data['title'].lower().split())

        # Remove common words that don't indicate topic match
        stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'at',
                      'to', 'for', 'and', 'or', 'is', 'are', 'was', 'were'}
        topic_words = topic_words - stop_words
        title_words = title_words - stop_words

        if topic_words:
            overlap = len(topic_words & title_words)
            match_ratio = overlap / len(topic_words)
        else:
            match_ratio = 0

        # Also check for exact match (case-insensitive)
        exact_match = topic.lower().strip(
        ) == wiki_data['title'].lower().strip()

        if exact_match or match_ratio >= 0.7:
            use_wikipedia = True
            status_log += f"🎯 Found: '{wiki_data['title']}'\n\n"
        else:
            status_log += f"⚠️ No exact match found\n\n"
    else:
        status_log += "⚠️ No direct source found\n\n"

    yield article_placeholder, status_log, loading_message, None

    # Path A: Use Wikipedia (fast path)
    if use_wikipedia and wiki_data:
        # Set up source display
        streaming_sources_display = "📚 SOURCE MATERIAL\n"
        streaming_sources_display += "=" * 44 + "\n\n"
        streaming_sources_display += f"**{wiki_data['title']}**\n\n"
        streaming_sources_display += f"Source: {wiki_data['url']}\n\n"
        streaming_sources_display += "---\n\n"
        streaming_sources_display += f"{wiki_data['content'][:2500]}"

        source_context = f"Wikipedia Article: {wiki_data['title']}\n\n{wiki_data['content'][:3000]}"
        source_notes.append(
            f"[Wikipedia: {wiki_data['title']}]({wiki_data['url']})")
        current_sources = [{"name": wiki_data['title'], "type": "Wikipedia",
                            "url": wiki_data['url'], "content": wiki_data['content']}]

        status_log += "📝 Generating article...\n\n"
        yield article_placeholder, status_log, streaming_sources_display, None

        try:
            wiki_chat = client.chat.create(model="grok-4-1-fast")

            wiki_chat.append(system(
                "You are an expert knowledge synthesizer focused on epistemic integrity. "
                "Write factual, well-sourced articles with inline citations. "
                "Be clear about certainty levels and avoid speculation."
            ))

            wiki_prompt = f"""Write a comprehensive, factual article about "{topic}".

Use the following Wikipedia content as your primary source:

{source_context}

Requirements:
1. Write in encyclopedic style (factual, neutral, well-structured)
2. Include inline citations like [1] when referencing facts
3. Add a "## Sources" section at the end listing all references
4. Be clear about certainty levels - use phrases like "evidence suggests", "widely accepted", etc.
5. Length: EXACTLY 300 words (excluding the Sources section)
6. Structure: Introduction, 2-3 main sections, conclusion, sources

Format the article in clean markdown."""

            wiki_chat.append(user(wiki_prompt))

            article_content = ""
            for response, chunk in wiki_chat.stream():
                if chunk.content:
                    article_content += chunk.content

                    streaming_article = "📝 YOUR ARTICLE\n"
                    streaming_article += "=" * 44 + "\n\n"
                    streaming_article += article_content

                    yield streaming_article, status_log, streaming_sources_display, None

        except Exception as e:
            status_log += f"❌ Error generating article: {str(e)}\n\n"
            error_article = "📝 YOUR ARTICLE\n" + "=" * 44 + \
                "\n\n❌ Error generating article. Please try again."
            yield error_article, status_log, streaming_sources_display, None
            return

    # Path B: Use web search (slower path for specific/niche topics)
    else:
        status_log += "🌐 Using web search for specific content...\n\n"
        yield article_placeholder, status_log, loading_message, None

        try:
            chat = client.chat.create(
                model="grok-4-1-fast",
                include=["inline_citations"],
                tools=[web_search()],
            )

            chat.append(system(
                "You are an expert knowledge synthesizer focused on epistemic integrity. "
                "Search the web for ONE authoritative source about the topic, then write a factual, "
                "well-sourced article with inline citations. Use exactly 1 high-quality source. "
                "Be clear about certainty levels and avoid speculation."
            ))

            prompt = f"""Search the web for authoritative information about "{topic}", then write a comprehensive, factual article.

Requirements:
1. Search for and use exactly 1 high-quality, authoritative source (prefer academic sites, reputable news, official sources)
2. Write in encyclopedic style (factual, neutral, well-structured)
3. Include inline citations referencing your source
4. Add a "## Sources" section at the end listing the reference with URL
5. Be clear about certainty levels - use phrases like "evidence suggests", "widely accepted", etc.
6. Length: EXACTLY 300 words (excluding the Sources section)
7. Structure: Introduction, 2-3 main sections, conclusion, sources

Format the article in clean markdown."""

            chat.append(user(prompt))

            status_log += "📡 Searching web sources...\n\n"
            yield article_placeholder, status_log, loading_message, None

            import time
            article_content = ""
            final_response = None
            first_chunk = True
            start_time = time.time()
            last_update_time = start_time
            update_messages = [
                (5, "🔍 Finding relevant sources..."),
                (10, "📖 Reading source content..."),
                (15, "✨ Preparing article structure..."),
            ]
            next_update_idx = 0

            for response, chunk in chat.stream():
                final_response = response

                # Time-based updates while waiting for content
                elapsed = time.time() - start_time
                if first_chunk and next_update_idx < len(update_messages):
                    threshold, message = update_messages[next_update_idx]
                    if elapsed >= threshold:
                        status_log += f"{message}\n\n"
                        yield article_placeholder, status_log, loading_message, None
                        next_update_idx += 1

                if chunk.content:
                    if first_chunk:
                        status_log += "📝 Writing article...\n\n"
                        first_chunk = False

                    article_content += chunk.content
                    streaming_article = "📝 YOUR ARTICLE\n"
                    streaming_article += "=" * 44 + "\n\n"
                    streaming_article += article_content
                    yield streaming_article, status_log, loading_message, None

            # Extract cited URLs
            import re
            cited_urls = re.findall(
                r'\]\((https?://[^\)]+)\)', article_content)
            seen = set()
            unique_cited_urls = []
            for url in cited_urls:
                if url not in seen:
                    seen.add(url)
                    unique_cited_urls.append(url)

            citations = unique_cited_urls[:1] if unique_cited_urls else []
            if not citations and final_response and hasattr(final_response, 'citations') and final_response.citations:
                citations = list(final_response.citations)[:1]

            if citations:
                status_log += "📄 Processing source...\n\n"
                yield streaming_article, status_log, loading_message, None

                url = citations[0]
                source_data = fetch_content_from_url(url)

                streaming_sources_display = "📚 WEB SOURCE\n"
                streaming_sources_display += "=" * 44 + "\n\n"

                if source_data and source_data.get("content"):
                    content = source_data["content"]
                    words = content.split()
                    truncated_content = " ".join(words[:250])
                    if len(words) > 250:
                        truncated_content += "..."

                    streaming_sources_display += f"Source: {url}\n\n"
                    streaming_sources_display += "-" * 44 + "\n\n"
                    streaming_sources_display += truncated_content

                    source_notes.append(f"[{source_data['title']}]({url})")
                    current_sources.append({
                        "name": source_data["title"],
                        "type": "Web",
                        "url": url,
                        "content": content
                    })
                    source_context = f"Web Source: {source_data['title']}\n\n{content[:3000]}"
                else:
                    streaming_sources_display += f"**{url}**\n\n"
                    streaming_sources_display += "(Content preview not available for this source)"
                    source_notes.append(f"[Source]({url})")
                    current_sources.append(
                        {"name": url, "type": "Web", "url": url})
                    source_context = f"Web Source: {url}"
            else:
                status_log += "⚠️ No citations found in response\n\n"
                streaming_sources_display = "📚 SOURCE MATERIAL\n" + "=" * 44 + \
                    "\n\n⚠️ No sources found.\n\nArticle generated from general knowledge."

        except Exception as e:
            status_log += f"❌ Web search failed: {str(e)}\n\n"
            error_article = "📝 YOUR ARTICLE\n" + "=" * 44 + \
                "\n\n❌ Error generating article. Please try again."
            yield error_article, status_log, loading_message, None
            return

    # Add header at the top
    final_article = "📝 YOUR ARTICLE\n"
    final_article += "=" * 44 + "\n\n"
    final_article += article_content

    # Store in history
    article_history.append(final_article)

    # Store clean article and grounding context for debates
    current_article_clean = article_content
    original_article = article_content
    current_grounding_context = source_context

    # Update status log to show completion
    status_log += "✅ Article generation complete!\n\n"

    # Save article to file for download
    import re
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[<>:"/\\|?*]', '', topic)
    safe_topic = safe_topic.replace(' ', '_')[:50]
    filename = f"article_{safe_topic}_{timestamp}.md"
    filepath = os.path.abspath(filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(article_content)
    except Exception as e:
        print(f"Error saving article: {e}")
        filepath = None

    # Yield final result with filepath for download
    yield final_article, status_log, streaming_sources_display, filepath


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

with gr.Blocks(theme=dark_theme, title="Veritas Epistemics - Truth-Seeking Article Generator", fill_height=False) as demo:
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
            padding: 0 0px 20px 0px !important;
            align-items: flex-start !important;
            border: none !important;
            background: transparent !important;
            height: auto !important;
            min-height: auto !important;
        }

        /* Target the block containers that wrap each panel - THIS IS THE CULPRIT */
        .article-row .block,
        .article-row .block.side-panel,
        .article-row .block.left-panel,
        .article-row .block.right-panel,
        .article-row .block.central-article {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
        }

        /* Kill the unequal-height class that adds extra space */
        .article-row.unequal-height,
        .row.unequal-height {
            height: auto !important;
            min-height: auto !important;
            max-height: fit-content !important;
        }

        /* Target the main fillable container - this is what fills viewport */
        main {
            min-height: 0 !important;
            height: auto !important;
        }

        main.fillable,
        main.app {
            min-height: 0 !important;
            height: auto !important;
        }

        /* Kill any space after article row */
        .article-row ~ * {
            display: none !important;
        }

        /* Ensure container height fits content */
        .gradio-container {
            max-height: fit-content !important;
            height: auto !important;
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
            height: auto !important;
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
            height: auto !important;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
            margin-top: 0 !important;
        }

        /* Force central article label to have no top offset */
        label.central-article {
            margin-top: 0 !important;
        }

        /* Make textareas scrollable with reasonable fixed height */
        .side-panel textarea,
        .left-panel textarea,
        .right-panel textarea,
        .central-article textarea {
            height: 600px !important;
            min-height: 600px !important;
            max-height: 600px !important;
            resize: none !important;
            overflow-y: scroll !important;
            overflow-x: hidden !important;
            scrollbar-width: thin !important;
            scrollbar-color: #6366f1 #0f0f0f !important;
        }

        /* Chrome/Safari/Edge scrollbar styling - make it VERY visible */
        .side-panel textarea::-webkit-scrollbar,
        .left-panel textarea::-webkit-scrollbar,
        .right-panel textarea::-webkit-scrollbar,
        .central-article textarea::-webkit-scrollbar {
            width: 16px !important;
            display: block !important;
        }

        .side-panel textarea::-webkit-scrollbar-track,
        .left-panel textarea::-webkit-scrollbar-track,
        .right-panel textarea::-webkit-scrollbar-track,
        .central-article textarea::-webkit-scrollbar-track {
            background: #2a2a2a !important;
            border-radius: 0px !important;
        }

        .side-panel textarea::-webkit-scrollbar-thumb,
        .left-panel textarea::-webkit-scrollbar-thumb,
        .right-panel textarea::-webkit-scrollbar-thumb,
        .central-article textarea::-webkit-scrollbar-thumb {
            background: #6366f1 !important;
            border-radius: 0px !important;
            border: none !important;
        }

        .side-panel textarea::-webkit-scrollbar-thumb:hover,
        .left-panel textarea::-webkit-scrollbar-thumb:hover,
        .right-panel textarea::-webkit-scrollbar-thumb:hover,
        .central-article textarea::-webkit-scrollbar-thumb:hover {
            background: #818cf8 !important;
        }

        .central-article::placeholder {
            color: #777777 !important;
            opacity: 0.7 !important;
            font-style: italic !important;
            text-align: center !important;
        }

        /* Synthetic controls panel specific styling */
        #synthetic-controls-panel {
            border-radius: 4px !important;
            background-color: #111111 !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
            padding: 10px 16px 16px 16px !important;
            margin-top: 16px !important;
            margin-left: 12px !important;
            min-height: 600px !important;
            font-family: monospace !important;
        }

        #synthetic-controls-panel h3 {
            color: #e5e7eb !important;
            margin-top: 0 !important;
            font-family: monospace !important;
            font-weight: 300 !important;
        }

        #synthetic-controls-panel label {
            font-family: monospace !important;
        }

        /* Move Generation Controls text up - using position relative */
        #synthetic-controls-panel .gr-markdown,
        #synthetic-controls-panel .markdown,
        #synthetic-controls-panel div.markdown {
            position: relative !important;
            top: -10px !important;
            margin-bottom: -10px !important;
        }

        /* Move the first divider line up */
        #synthetic-controls-panel > div:nth-child(2) {
            margin-top: -21px !important;
        }

        /* Reduce gap between divider lines and controls below them */
        #divider-1, #divider-2, #divider-3, #divider-4 {
            margin-bottom: -10px !important;
        }

        /* Ensure all controls are left-aligned */
        #synthetic-controls-panel > div {
            margin-left: 0 !important;
            padding-left: 0 !important;
            align-items: flex-start !important;
        }

        /* Force all form controls to left align */
        #synthetic-controls-panel .block.gr-number,
        #synthetic-controls-panel .block.gr-dropdown {
            margin-left: 0 !important;
            padding-left: 0 !important;
        }

        /* Reduce spacing between radio buttons */
        #synthetic-controls-panel .wrap {
            gap: 0.5px !important;
        }

        #synthetic-controls-panel label {
            margin-bottom: 0.5px !important;
            margin-left: 0 !important;
        }

        /* Remove focus glow from number input */
        #synthetic-controls-panel input[type="number"]:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* Style quality dropdown with visible border - only button not label */
        #quality-dropdown > div > div:last-child,
        #quality-dropdown button {
            border: 1px solid #666666 !important;
            border-radius: 4px !important;
        }

        /* Style flaw dropdown with visible border - only button not label */
        #flaw-dropdown > div > div:last-child,
        #flaw-dropdown button {
            border: 1px solid #666666 !important;
            border-radius: 4px !important;
        }

        /* Style length dropdown with visible border - only button not label */
        #length-dropdown > div > div:last-child,
        #length-dropdown button {
            border: 1px solid #666666 !important;
            border-radius: 4px !important;
        }

        /* Version History Panel - raw HTML injection */
        #version-panel {
            position: fixed;
            top: 0;
            right: -100%;
            width: 90%;
            height: 100vh;
            background-color: #0f0f0f;
            border-left: 3px solid #ffffff;
            z-index: 9999;
            overflow: hidden;
            padding: 20px;
            box-sizing: border-box;
            box-shadow: -4px 0 20px rgba(0, 0, 0, 0.5);
            transition: right 0.5s ease;
            pointer-events: none;
            display: flex;
            flex-direction: column;
        }

        #version-panel.visible {
            right: 0;
            pointer-events: auto;
        }

        #version-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.2rem;
            font-weight: bold;
            color: #6366f1;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #333;
            flex-shrink: 0;
        }

        #close-version-panel:hover {
            color: #f87171;
            transform: scale(1.2);
        }

        #version-panel-content {
            display: flex;
            flex: 1;
            gap: 20px;
            overflow: hidden;
        }

        #version-preview {
            flex: 1;
            background-color: #1a1a1a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 0.9rem;
            color: #e5e7eb;
            line-height: 1.6;
        }

        #version-list-container {
            width: 120px;
            flex-shrink: 0;
            overflow-y: auto;
        }

        .version-item {
            background-color: #1a1a1a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 4px 8px;
            margin-bottom: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: monospace;
        }

        .version-item:hover {
            border-color: #ffffff;
            background-color: #252525;
        }

        .version-item.latest {
            /* No special border - only .selected gets white border */
        }

        .version-item.selected {
            border-color: #ffffff;
            background-color: #252525;
        }

        .version-label {
            font-weight: normal;
            font-size: 0.8rem;
            color: #e5e7eb;
            margin-bottom: 1px;
            line-height: 1.2;
        }

        .version-type {
            font-size: 0.8rem;
            color: #9ca3af;
            line-height: 1.2;
        }

        .version-history-header {
            font-size: 1.2rem !important;
            font-weight: bold !important;
            color: #6366f1 !important;
            margin-bottom: 20px !important;
            padding-bottom: 10px !important;
            border-bottom: 2px solid #333 !important;
        }

        .version-item {
            background-color: #1a1a1a !important;
            border: 2px solid #333 !important;
            border-radius: 8px !important;
            padding: 4px 8px !important;
            margin-bottom: 4px !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            font-family: monospace !important;
        }

        .version-item:hover {
            border-color: #ffffff !important;
            background-color: #252525 !important;
        }

        .version-item.latest {
            /* No special border - only .selected gets white border */
        }

        .version-item.selected {
            border-color: #ffffff !important;
            background-color: #252525 !important;
        }

        .version-label {
            font-weight: normal !important;
            font-size: 0.8rem !important;
            color: #e5e7eb !important;
            margin-bottom: 1px !important;
            line-height: 1.2 !important;
        }

        .version-type {
            font-size: 0.8rem !important;
            color: #9ca3af !important;
            line-height: 1.2 !important;
        }

        #restore-version-btn:hover:not(:disabled) {
            border-color: #ffffff !important;
            color: #ffffff !important;
            background-color: #252525 !important;
        }

        #version-history-btn {
            background-color: #0f0f0f !important;
            border: 1.5px solid #e5e7eb !important;
            color: #e5e7eb !important;
            border-radius: 8px !important;
            font-family: monospace !important;
            font-size: 0.9rem !important;
            padding: 8px 10px !important;
            height: 38px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }

        #version-history-btn:hover {
            background-color: #1a1a1a !important;
            border-color: #6366f1 !important;
            color: #6366f1 !important;
        }

        #download-btn {
            background-color: #0f0f0f !important;
            border: 1.5px solid #e5e7eb !important;
            color: #e5e7eb !important;
            border-radius: 8px !important;
            font-family: monospace !important;
            font-size: 0.9rem !important;
            padding: 8px 10px !important;
            height: 38px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }

        #download-btn:hover {
            background-color: #1a1a1a !important;
            border-color: #6366f1 !important;
            color: #6366f1 !important;
        }

        #download-btn[disabled],
        #download-btn.disabled {
            cursor: not-allowed !important;
            opacity: 0.4 !important;
            border-color: #444444 !important;
            color: #666666 !important;
        }

        #download-btn[disabled]:hover,
        #download-btn.disabled:hover {
            background-color: #0f0f0f !important;
            border-color: #444444 !important;
            color: #666666 !important;
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

        // Inject version panel directly into body on page load
        function injectVersionPanel() {
            if (!document.getElementById('version-panel')) {
                const panel = document.createElement('div');
                panel.id = 'version-panel';
                panel.innerHTML = `
                    <div id="version-panel-header" style="display: flex; justify-content: space-between; align-items: center; font-size: 1.2rem; font-weight: normal; color: #ffffff; font-family: monospace; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #333; flex-shrink: 0;">
                        <span>⏳ VERSION HISTORY</span>
                        <button id="close-version-panel" onclick="document.getElementById('version-panel').classList.remove('visible')" style="background: none; border: none; color: #ffffff; font-size: 1.5rem; cursor: pointer; padding: 0; line-height: 1;">&times;</button>
                    </div>
                    <div id="version-panel-content" style="display: flex; flex-direction: row; flex: 1; gap: 20px; overflow: hidden; min-height: 0; margin-bottom: 15px;">
                        <div id="version-preview-container" style="flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; gap: 10px;">
                            <div id="version-preview" style="flex: 1; min-height: 0; background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px; padding: 20px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; color: #e5e7eb; line-height: 1.6; box-sizing: border-box;">Select a version to preview</div>
                            <button id="restore-version-btn" onclick="restoreSelectedVersion()" disabled style="padding: 8px 16px; background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px; color: #555; font-family: monospace; font-size: 0.85rem; cursor: not-allowed; transition: all 0.2s ease; width: 100%; opacity: 0.5;">Restore This Version</button>
                        </div>
                        <div id="version-list-container" style="width: 120px; min-width: 120px; flex-shrink: 0; overflow-y: auto;">
                            <div id="version-list">
                                <div style='color: #9ca3af; text-align: center; padding: 20px; font-size: 0.7rem;'>No versions yet.</div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.appendChild(panel);
            }
        }

        // Try multiple times to ensure it loads
        window.addEventListener('DOMContentLoaded', injectVersionPanel);
        window.addEventListener('load', injectVersionPanel);
        setTimeout(injectVersionPanel, 100);
        setTimeout(injectVersionPanel, 500);

        // Initialize global storage for version articles
        window.versionArticles = [];
        window.selectedVersionNum = null;

        // Function called when clicking a version item
        window.selectVersion = function(versionNum) {
            console.log('selectVersion called with:', versionNum);
            console.log('Available articles:', window.versionArticles.length);
            const articles = window.versionArticles || [];
            const article = articles.find(a => a.version === versionNum);
            console.log('Found article:', article ? 'yes' : 'no');
            if (article) {
                window.selectedVersionNum = versionNum;
                const preview = document.getElementById('version-preview');
                if (preview) {
                    preview.textContent = article.content;
                    console.log('Preview updated');
                }
                // Update selected state
                document.querySelectorAll('.version-item').forEach(item => {
                    item.classList.remove('selected');
                    if (item.dataset.version == versionNum) {
                        item.classList.add('selected');
                    }
                });
            }
        };

        // Function to restore the selected version
        window.restoreSelectedVersion = function() {
            if (window.selectedVersionNum === null) {
                console.log('No version selected');
                return;
            }
            const articles = window.versionArticles || [];
            const article = articles.find(a => a.version === window.selectedVersionNum);
            if (article) {
                // Set the content in the hidden textbox
                console.log('Attempting to restore article, content length:', article.content.length);

                // Find the hidden input - try multiple selectors
                let hiddenInput = document.querySelector('#restore-version-input textarea');
                if (!hiddenInput) {
                    hiddenInput = document.querySelector('#restore-version-input input');
                }
                if (!hiddenInput) {
                    // Try finding any input inside the component
                    const wrapper = document.getElementById('restore-version-input');
                    if (wrapper) {
                        hiddenInput = wrapper.querySelector('textarea, input');
                    }
                }

                if (hiddenInput) {
                    hiddenInput.value = 'VERSION:' + window.selectedVersionNum + '|||' + article.content;
                    hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
                }

                setTimeout(() => {
                    const triggerBtn = document.querySelector('button#restore-trigger-btn');
                    if (triggerBtn) {
                        triggerBtn.click();
                    }
                    const panel = document.getElementById('version-panel');
                    if (panel) { panel.classList.remove('visible'); }
                    setTimeout(() => {
                        const toast = document.createElement('div');
                        toast.textContent = 'Article Restored!';
                        toast.style.cssText = 'position: fixed !important; top: 53% !important; left: 50% !important; transform: translate(-50%, -50%) !important; background-color: #1a1a1a !important; color: #fff !important; padding: 14px 24px !important; border-radius: 8px !important; border: 2px solid #fff !important; font-family: monospace !important; font-size: 0.9rem !important; z-index: 10000 !important; opacity: 0; transition: opacity 0.3s ease !important;';
                        document.body.appendChild(toast);
                        setTimeout(() => { toast.style.opacity = '1'; }, 10);
                        setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => { toast.remove(); }, 300); }, 4000);
                    }, 400);
                }, 100);
            }
        };
    </script>
    <style>
        /* Hidden offscreen but still functional for Gradio events */
        .hidden-offscreen {
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
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

        # Topic input (no button inside - action button handles execution)
        topic_input = gr.Textbox(
            placeholder="Enter a topic of your choice. e.g. Machine Learning, Astrology, Global Warming",
            lines=1,
            container=False,
            elem_id="topic-input-box",
            scale=1
        )

        # Dynamic action button
        action_button = gr.Button(
            value="Generate Article",
            variant="primary",
            scale=0,
            min_width=180,
            elem_id="action-button"
        )

        # Download button
        download_btn = gr.Button(
            value="🡳",
            variant="secondary",
            scale=0,
            min_width=55,
            elem_id="download-btn",
            interactive=False
        )

        # Version History button
        version_history_btn = gr.Button(
            value="⌛",
            variant="secondary",
            scale=0,
            min_width=55,
            elem_id="version-history-btn"
        )

    # The article panels row - now full width!
    # All 3 panels always visible to keep center article perfectly centered
    with gr.Row(elem_classes=["article-row"]):
        # Synthetic Data left panel (Column with controls + log) - only visible for Synthetic Data
        with gr.Column(scale=1, visible=False, elem_id="synthetic-controls-panel") as synthetic_left_panel:
            gr.Markdown("⚙️ GENERATION CONTROLS")

            gr.Markdown("=" * 36, elem_id="divider-1")

            num_examples_number = gr.Number(
                value=5,
                minimum=1,
                maximum=10,
                label="Number of Examples",
                info="Enter a number between 1-10"
            )

            gr.Markdown("=" * 36, elem_id="divider-2")

            quality_dropdown = gr.Dropdown(
                choices=["All", "Excellent", "Good", "Poor", "Terrible"],
                value="All",
                label="Quality Distribution",
                info="Target epistemic quality",
                elem_id="quality-dropdown"
            )

            gr.Markdown("=" * 36, elem_id="divider-3")

            flaw_dropdown = gr.Dropdown(
                choices=["Auto", "Citations", "Certainty", "Bias", "Multiple"],
                value="Auto",
                label="Flaw Type",
                info="Inject specific epistemic issue",
                elem_id="flaw-dropdown"
            )

            gr.Markdown("=" * 36, elem_id="divider-4")

            length_dropdown = gr.Dropdown(
                choices=["Brief (75-100)", "Standard (175-200)",
                         "Long (275-300)"],
                value="Standard (175-200)",
                label="Article Length",
                info="Target word count range",
                elem_id="length-dropdown"
            )

        # Hidden placeholder for synthetic_log (removed from display but needed for outputs)
        synthetic_log = gr.Textbox(visible=False)

        # Normal left panel - Process log and status (visible for all tools EXCEPT Synthetic Data)
        left_panel = gr.Textbox(
            value="🔍 PROCESS LOG\n" + "=" * 42 +
            "\n\nThis panel will show progress updates like:\n• Searching for sources\n• Generating article\n• Completion status\n\nEnter a topic and click Generate Article to begin!",
            lines=30,
            interactive=False,
            show_copy_button=False,
            show_label=False,
            container=True,
            elem_classes=["side-panel", "left-panel"],
            visible=True
        )

        # Central article (always visible and centered!)
        article_display = gr.Textbox(
            value="📝 YOUR ARTICLE\n" + "=" * 42 +
            "\n\nYour generated article will appear here.\n\nKnowledge incoming!",
            lines=30,
            interactive=False,
            show_copy_button=False,
            container=True,
            elem_classes=["side-panel", "central-article"],
            show_label=False
        )

        # Right panel - Source material (Web Sources)
        right_panel = gr.Textbox(
            value="📚 SOURCE MATERIAL\n" + "=" * 42 +
            "\n\nThis panel will show sources like:\n• Web articles\n• Reference pages\n\nSources appear after generation.",
            lines=30,
            interactive=False,
            show_copy_button=False,
            show_label=False,
            container=True,
            elem_classes=["side-panel", "right-panel"],
            visible=True
        )

    # Hidden HTML component for version panel (must be HTML, not State, to pass to JS)
    version_state = gr.HTML(value="", visible=False,
                            elem_id="version-state-holder")

    # State to store download file path
    download_file_state = gr.State(value=None)

    # Hidden File component for downloads
    download_file = gr.File(visible=False, interactive=False)

    # Hidden components for restore version functionality (visible but hidden via CSS for Gradio compatibility)
    restore_version_input = gr.Textbox(
        value="", visible=True, elem_id="restore-version-input", elem_classes=["hidden-offscreen"])
    restore_trigger_btn = gr.Button(
        "Restore", visible=True, elem_id="restore-trigger-btn", elem_classes=["hidden-offscreen"])

    # Route action button to correct function based on dropdown
    def execute_action(selected_tool, topic, user_feedback, num_examples, quality_dist, flaw_type, article_length):
        if selected_tool == "Article Generation":
            # Generate article - yields to (center, left, right, filepath)
            for center, left, right, filepath in generate_initial_article(topic):
                download_enabled = filepath is not None
                yield center, left, left, right, filepath, gr.update(interactive=download_enabled)
        elif selected_tool == "Multi-Agent Debate":
            # Run debate - yields (left=transcript, center=article, right=edit_log, filepath)
            for left, center, right, filepath in run_multi_agent_debate():
                download_enabled = filepath is not None
                yield center, left, left, right, filepath, gr.update(interactive=download_enabled)
        elif selected_tool == "Self-Critique":
            # Run self-critique - yields (left=critique, center=article, right=edit_log, filepath)
            for left, center, right, filepath in run_self_critique():
                download_enabled = filepath is not None
                yield center, left, left, right, filepath, gr.update(interactive=download_enabled)
        elif selected_tool == "User Feedback":
            # Process user feedback - yields (center, left, right, filepath)
            for center, left, right, filepath in run_user_feedback(user_feedback):
                download_enabled = filepath is not None
                yield center, left, left, right, filepath, gr.update(interactive=download_enabled)
        elif selected_tool == "Synthetic Data":
            # Generate synthetic training data - yields to (center, synthetic_log, right, filename)
            for center, synth_log, right, filename in run_synthetic_data_generation(topic, num_examples, quality_dist, flaw_type, article_length):
                # Enable download button when filename is available
                download_enabled = filename is not None
                # Empty string for left_panel, actual log for synthetic_log
                yield center, "", synth_log, right, filename, gr.update(interactive=download_enabled)
        else:
            # Placeholder for other tools
            error_msg = f"⚠️ {selected_tool} not yet implemented."
            yield "", error_msg, error_msg, "", None, gr.update(interactive=False)

    # Update action button text and input state based on dropdown selection
    def update_ui_state(selected_tool, topic):
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

        # Disable topic input when epistemic tool is selected (except Synthetic Data which allows any topic)
        topic_disabled = (selected_tool not in [
                          "Article Generation", "Synthetic Data"])

        # For Synthetic Data, disable button if topic is empty
        if selected_tool == "Synthetic Data":
            button_disabled = not topic or topic.strip() == ""
        else:
            button_disabled = topic_disabled and not current_article_clean

        # Set placeholder content for panels based on selected tool
        if selected_tool == "Article Generation":
            left_placeholder = "🔍 PROCESS LOG\n" + "=" * 42 + \
                "\n\nThis panel will show progress updates like:\n• Searching for sources\n• Generating article\n• Completion status\n\nEnter a topic and click Generate Article to begin!"
            center_placeholder = "📝 YOUR ARTICLE\n" + "=" * 42 + \
                "\n\nYour generated article will appear here.\n\nKnowledge incoming!"
            right_placeholder = "📚 SOURCE MATERIAL\n" + "=" * 42 + \
                "\n\nThis panel will show sources like:\n• Web articles\n• Reference pages\n\nSources appear after generation."

        elif selected_tool == "Self-Critique":
            left_placeholder = "💭 SELF-CRITIQUE ANALYSIS\n" + "=" * 42 + \
                "\n\nThis panel will show:\n• Epistemic quality assessment\n• Identification of issues\n• Suggestions for improvement\n\nClick 'Critique Article' to begin!"

            # If article exists, show it with the "ORIGINAL ARTICLE" header
            if current_article_clean:
                center_placeholder = "📝 ORIGINAL ARTICLE\n" + \
                    "=" * 42 + "\n\n" + current_article_clean
            else:
                center_placeholder = "📝 ORIGINAL ARTICLE\n" + "=" * 42 + \
                    "\n\nYour article will appear here.\n\nAfter the critique, this will show the revised version."

            right_placeholder = "📝 EDIT LOG\n" + "=" * 42 + \
                "\n\nThis panel will show what changed:\n• Substitutions\n• Additions\n• Deletions\n\nEdit log appears after critique completes."

        elif selected_tool == "Multi-Agent Debate":
            left_placeholder = "🎭 DEBATE TRANSCRIPT\n" + "=" * 42 + \
                "\n\nThis panel will show the debate between:\n• Defender: Argues for strengths\n• Challenger: Identifies weaknesses\n• Arbiter: Synthesizes improvements\n\nClick 'Start Debate' to begin!"

            # If article exists, show it with the "ORIGINAL ARTICLE" header
            if current_article_clean:
                center_placeholder = "📄 ORIGINAL ARTICLE\n" + \
                    "=" * 42 + "\n\n" + current_article_clean
            else:
                center_placeholder = "📄 ORIGINAL ARTICLE\n" + "=" * 42 + \
                    "\n\nYour article will appear here.\n\nAfter the debate, this will show the revised version."

            right_placeholder = "📝 EDIT LOG\n" + "=" * 42 + \
                "\n\nThis panel will show what changed:\n• Substitutions\n• Additions\n• Deletions\n\nEdit log appears after debate completes."

        elif selected_tool == "User Feedback":
            # Left panel becomes interactive for user input - use placeholder instead of value
            left_panel_placeholder = "Enter your feedback here...\n\nSuggest improvements to:\n- Claims needing qualifiers or sources\n- Overstatements or unwarranted certainty\n- Missing perspectives or caveats\n- Structural or clarity issues\n\nBe specific and reference particular sections."
            left_placeholder = ""  # Empty value, will use placeholder instead

            # If article exists, show it in center
            if current_article_clean:
                center_placeholder = "📝 CURRENT ARTICLE\n" + \
                    "=" * 42 + "\n\n" + current_article_clean
            else:
                center_placeholder = "📝 CURRENT ARTICLE\n" + "=" * 42 + \
                    "\n\nYour article will appear here.\n\nAfter the feedback, this will show the revised version."

            right_placeholder = "📋 CHANGELOG\n" + "=" * 42 + "\n\nThis panel will display the changelog after processing your feedback:\n\n✅ ACCEPTED changes\n   - What was incorporated and why\n\n⚠️ PARTIALLY ACCEPTED changes\n   - What was modified and reasoning\n\n❌ REJECTED changes\n   - Why suggestions weren't incorporated\n\nThe revised article will appear in the center panel.\n\nEnter your feedback in the left panel and click 'Collect Feedback'."

        elif selected_tool == "Synthetic Data":
            # Synthetic Data Generation placeholders
            left_placeholder = "🏭 GENERATION LOG\n" + "=" * 42 + "\n\nThis panel displays the synthetic data generation process:\n\n- Configuration details\n- Generation progress for each example\n- Quality tier assignments\n- Epistemic flaw injection\n- Export status and file location\n\nEnter a topic and click 'Generate Data' to create labeled training examples!"

            center_placeholder = "📝 ARTICLE PREVIEW\n" + "=" * 42 + "\n\nThis panel will display generated articles as they are created:\n\n- Each article with its target quality level\n- Controlled epistemic characteristics\n- Varied certainty language patterns\n- Different sourcing qualities\n\nGenerated articles will appear here during the process."

            right_placeholder = "📋 METADATA & LABELS\n" + "=" * 42 + \
                "\n\nThis panel displays structured metadata for each generated example:\n\n- Epistemic quality scores (0-10)\n- Identified flaws and issues\n- Target quality tier\n- Dataset summary statistics\n- Export file information\n\nLabeled data suitable for training classifiers will appear here."

        else:
            # Default placeholders for other tools
            left_placeholder = "🔍 PROCESS LOG\n" + "=" * 42 + \
                "\n\nProcess information will appear here."
            left_panel_placeholder = None  # No placeholder for other tools
            center_placeholder = "📝 YOUR ARTICLE\n" + "=" * \
                42 + "\n\nYour article content will appear here."
            right_placeholder = "📚 OUTPUT\n" + "=" * 42 + "\n\nResults will appear here."

        # Determine if left panel should be interactive
        left_interactive = (selected_tool == "User Feedback")

        # Determine panel visibility based on tool
        show_synthetic_panel = (selected_tool == "Synthetic Data")
        show_normal_left = not show_synthetic_panel

        # Disable download button when switching away from Synthetic Data
        # Always disabled when switching tools (will be enabled after generation completes)
        download_btn_enabled = False

        # For User Feedback, clear value and use placeholder; for others, use value
        if selected_tool == "User Feedback":
            return (
                gr.update(value=button_text, interactive=not button_disabled),
                gr.update(interactive=not topic_disabled),
                gr.update(value=center_placeholder),
                gr.update(value="", placeholder=left_panel_placeholder,
                          interactive=left_interactive, visible=show_normal_left),
                gr.update(value=right_placeholder),
                # synthetic_left_panel
                gr.update(visible=show_synthetic_panel),
                gr.update(interactive=download_btn_enabled)  # download_btn
            )
        else:
            return (
                gr.update(value=button_text, interactive=not button_disabled),
                gr.update(interactive=not topic_disabled),
                gr.update(value=center_placeholder),
                gr.update(value=left_placeholder,
                          interactive=left_interactive, visible=show_normal_left),
                gr.update(value=right_placeholder),
                # synthetic_left_panel
                gr.update(visible=show_synthetic_panel),
                gr.update(interactive=download_btn_enabled)  # download_btn
            )

    epistemic_dropdown.change(
        fn=update_ui_state,
        inputs=[epistemic_dropdown, topic_input],
        outputs=[action_button, topic_input,
                 article_display, left_panel, right_panel,
                 synthetic_left_panel, download_btn],
        show_progress="hidden"
    )

    # Update button state when topic changes (for Synthetic Data page)
    # Only update action button, don't re-render panels to prevent flashing
    def update_button_on_topic_change(selected_tool, topic):
        # Only enable/disable action button based on topic content
        if selected_tool == "Synthetic Data":
            button_disabled = not topic or topic.strip() == ""
            return gr.update(interactive=not button_disabled)
        return gr.update()

    topic_input.change(
        fn=update_button_on_topic_change,
        inputs=[epistemic_dropdown, topic_input],
        outputs=[action_button],
        show_progress="hidden"
    )

    action_button.click(
        fn=execute_action,
        inputs=[epistemic_dropdown, topic_input, left_panel,
                num_examples_number, quality_dropdown, flaw_dropdown, length_dropdown],
        outputs=[article_display, left_panel, synthetic_log,
                 right_panel, download_file_state, download_btn],
        show_progress="hidden"
    ).then(
        fn=update_version_history,
        inputs=[article_display, left_panel, right_panel],
        outputs=[version_state]
    ).then(
        fn=None,
        inputs=[version_state],
        js="""(versionData) => {
            setTimeout(() => {
                const textareas = document.querySelectorAll('textarea');
                textareas.forEach(t => { t.scrollTop = 0; });
            }, 100);

            // Parse the version data
            let data;
            try {
                data = typeof versionData === 'string' ? JSON.parse(versionData) : versionData;
            } catch(e) {
                data = { list: versionData, articles: [], latest_content: '' };
            }

            // Update global articles data
            window.versionArticles = data.articles || [];

            // Update the version list if panel exists
            const versionList = document.getElementById('version-list');
            if (versionList && data.list) {
                versionList.innerHTML = data.list;
            }

            // Update preview with latest content if panel is visible
            const panel = document.getElementById('version-panel');
            const preview = document.getElementById('version-preview');
            if (panel && panel.classList.contains('visible') && preview && data.latest_content) {
                preview.textContent = data.latest_content;
            }
        }"""
    )

    # Download button click handler
    def trigger_download(filepath):
        """Return the file path for download."""
        if filepath and filepath.strip():
            return filepath
        return None

    download_btn.click(
        fn=trigger_download,
        inputs=[download_file_state],
        outputs=[download_file]
    )

    # Auto-set flaw type to "Auto" and disable when quality is "Excellent"
    def update_flaw_type(quality_selection):
        if quality_selection == "Excellent":
            return gr.update(value="Auto", interactive=False)
        return gr.update(interactive=True)

    quality_dropdown.change(
        fn=update_flaw_type,
        inputs=[quality_dropdown],
        outputs=[flaw_dropdown]
    )

    # Version History button click handler
    version_history_btn.click(
        fn=toggle_version_panel,
        outputs=[version_state]
    ).then(
        fn=None,
        inputs=[version_state],
        js="""(versionData) => {
            // Parse the version data
            let data;
            try {
                data = typeof versionData === 'string' ? JSON.parse(versionData) : versionData;
                console.log('Version data parsed:', data);
                console.log('Articles count:', data.articles ? data.articles.length : 0);
                console.log('Latest content length:', data.latest_content ? data.latest_content.length : 0);
            } catch(e) {
                console.error('Failed to parse version data:', e);
                data = { list: versionData, articles: [], latest_content: '' };
            }

            // Store articles globally for selectVersion function
            window.versionArticles = data.articles || [];
            console.log('Stored versionArticles:', window.versionArticles.length);

            // Ensure panel exists
            if (!document.getElementById('version-panel')) {
                const panel = document.createElement('div');
                panel.id = 'version-panel';
                panel.innerHTML = `
                    <div id="version-panel-header" style="display: flex; justify-content: space-between; align-items: center; font-size: 1.2rem; font-weight: normal; color: #ffffff; font-family: monospace; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #333; flex-shrink: 0;">
                        <span>⏳ VERSION HISTORY</span>
                        <button id="close-version-panel" onclick="document.getElementById('version-panel').classList.remove('visible')" style="background: none; border: none; color: #ffffff; font-size: 1.5rem; cursor: pointer; padding: 0; line-height: 1;">&times;</button>
                    </div>
                    <div id="version-panel-content" style="display: flex; flex-direction: row; flex: 1; gap: 20px; overflow: hidden; min-height: 0; margin-bottom: 15px;">
                        <div id="version-preview-container" style="flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; gap: 10px;">
                            <div id="version-preview" style="flex: 1; min-height: 0; background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px; padding: 20px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; color: #e5e7eb; line-height: 1.6; box-sizing: border-box;">Select a version to preview</div>
                            <button id="restore-version-btn" onclick="restoreSelectedVersion()" disabled style="padding: 8px 16px; background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px; color: #555; font-family: monospace; font-size: 0.85rem; cursor: not-allowed; transition: all 0.2s ease; width: 100%; opacity: 0.5;">Restore This Version</button>
                        </div>
                        <div id="version-list-container" style="width: 120px; min-width: 120px; flex-shrink: 0; overflow-y: auto;">
                            <div id="version-list"></div>
                        </div>
                    </div>
                `;
                document.body.appendChild(panel);
            }

            // Update the version list content
            const versionList = document.getElementById('version-list');
            if (versionList && data.list) {
                versionList.innerHTML = data.list;
            }

            // Update preview with latest content
            const preview = document.getElementById('version-preview');
            if (preview && data.latest_content) {
                preview.textContent = data.latest_content;
            }

            // Enable/disable restore button (need at least 2 versions to restore)
            const restoreBtn = document.getElementById('restore-version-btn');
            if (restoreBtn) {
                if (data.articles && data.articles.length > 1) {
                    restoreBtn.disabled = false;
                    restoreBtn.style.opacity = '1';
                    restoreBtn.style.cursor = 'pointer';
                    restoreBtn.style.color = '#fff';
                } else {
                    restoreBtn.disabled = true;
                    restoreBtn.style.opacity = '0.5';
                    restoreBtn.style.cursor = 'not-allowed';
                    restoreBtn.style.color = '#555';
                }
            }

            // Toggle panel visibility
            const panel = document.getElementById('version-panel');
            if (panel) {
                panel.classList.toggle('visible');
            }

            // Define selectVersion function globally
            window.selectVersion = function(versionNum) {
                const articles = window.versionArticles || [];
                const article = articles.find(a => a.version === versionNum);
                if (article) {
                    window.selectedVersionNum = versionNum;
                    const preview = document.getElementById('version-preview');
                    if (preview) {
                        preview.textContent = article.content;
                    }
                    // Update selected state
                    document.querySelectorAll('.version-item').forEach(item => {
                        item.classList.remove('selected');
                        if (item.dataset.version == versionNum) {
                            item.classList.add('selected');
                        }
                    });
                }
            };

            // Set initial selected version to latest
            if (data.articles && data.articles.length > 0) {
                window.selectedVersionNum = data.articles[0].version;
            }

            // Define restoreSelectedVersion function globally
            window.restoreSelectedVersion = function() {
                if (window.selectedVersionNum === null) {
                    console.log('No version selected');
                    return;
                }
                const articles = window.versionArticles || [];
                const article = articles.find(a => a.version === window.selectedVersionNum);
                if (article) {
                    let hiddenInput = document.querySelector('#restore-version-input textarea');
                    if (!hiddenInput) {
                        const wrapper = document.getElementById('restore-version-input');
                        if (wrapper) { hiddenInput = wrapper.querySelector('textarea, input'); }
                    }
                    if (hiddenInput) {
                        hiddenInput.value = 'VERSION:' + window.selectedVersionNum + '|||' + article.content;
                        hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    setTimeout(() => {
                        const triggerBtn = document.querySelector('button#restore-trigger-btn');
                        if (triggerBtn) {
                            triggerBtn.click();
                        }
                        const panel = document.getElementById('version-panel');
                        if (panel) { panel.classList.remove('visible'); }
                        setTimeout(() => {
                            const toast = document.createElement('div');
                            toast.textContent = 'Article Restored!';
                            toast.style.cssText = 'position: fixed !important; top: 53% !important; left: 50% !important; transform: translate(-50%, -50%) !important; background-color: #1a1a1a !important; color: #fff !important; padding: 14px 24px !important; border-radius: 8px !important; border: 2px solid #fff !important; font-family: monospace !important; font-size: 0.9rem !important; z-index: 10000 !important; opacity: 0; transition: opacity 0.3s ease !important;';
                            document.body.appendChild(toast);
                            setTimeout(() => { toast.style.opacity = '1'; }, 10);
                            setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => { toast.remove(); }, 300); }, 4000);
                        }, 400);
                    }, 100);
                }
            };
        }"""
    )

    # Restore version button click handler
    def restore_version(content):
        global current_article_clean, article_history
        if content and content.strip():
            # Parse version number from input (format: VERSION:X|||CONTENT)
            version_num = None
            article_content = content.strip()

            if article_content.startswith("VERSION:") and "|||" in article_content:
                parts = article_content.split("|||", 1)
                version_num = parts[0].replace("VERSION:", "").strip()
                if len(parts) > 1:
                    article_content = parts[1]

            # Strip any existing headers
            if "=" * 20 in article_content:
                header_parts = article_content.split("=" * 20, 1)
                if len(header_parts) > 1:
                    article_content = header_parts[1].strip()
                    while article_content.startswith("="):
                        article_content = article_content[1:]
                    article_content = article_content.strip()

            current_article_clean = article_content

            # Format the restored article for display
            restored_display = "📝 RESTORED ARTICLE\n"
            restored_display += "=" * 44 + "\n\n"
            restored_display += current_article_clean

            # Add to article history with "Restored (from vX)" label
            if version_num:
                history_entry = f"📝 RESTORED ARTICLE (from v{version_num})\n"
            else:
                history_entry = "📝 RESTORED ARTICLE\n"
            history_entry += "=" * 44 + "\n\n"
            history_entry += current_article_clean
            article_history.append(history_entry)

            # Return placeholder values for all three panels
            left_placeholder = "🔍 PROCESS LOG\n" + "=" * 42 + \
                "\n\nThis panel will show progress updates like:\n• Searching for sources\n• Generating article\n• Completion status\n\nEnter a topic and click Generate Article to begin!"
            right_placeholder = "📚 SOURCE MATERIAL\n" + "=" * 42 + \
                "\n\nThis panel will show sources like:\n• Web articles\n• Reference pages\n\nSources appear after generation."

            return left_placeholder, restored_display, right_placeholder
        return gr.update(), gr.update(), gr.update()

    restore_trigger_btn.click(
        fn=restore_version,
        inputs=[restore_version_input],
        outputs=[left_panel, article_display, right_panel]
    )


# Launch with no footer
demo.launch(show_api=False)
