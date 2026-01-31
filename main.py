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
version_panel_visible = False  # Track if version history panel is visible


def toggle_source_view():
    """Toggle Wikipedia source panel content."""
    global source_visible, current_sources

    source_visible = not source_visible


def toggle_version_panel():
    """Toggle version history panel visibility."""
    global version_panel_visible
    version_panel_visible = not version_panel_visible
    return build_version_history_html()


def build_version_history_html():
    """Build HTML for version history list."""
    global article_history

    if not article_history:
        return "<div style='color: #9ca3af; text-align: center; padding: 20px;'>No versions yet. Generate an article to begin.</div>"

    html = ""
    for idx, article in enumerate(reversed(article_history)):
        version_num = len(article_history) - idx
        is_latest = (idx == 0)

        # Extract version type from article header
        if "Post-Debate" in article:
            version_type = "Post-Debate"
        elif "Post-Critique" in article:
            version_type = "Post-Critique"
        else:
            version_type = "Original"

        latest_class = " latest" if is_latest else ""

        html += f"""
        <div class='version-item{latest_class}' onclick='selectVersion({version_num - 1})'>
            <div class='version-label'>v{version_num} {"(Latest)" if is_latest else ""}</div>
            <div class='version-type'>{version_type}</div>
        </div>
        """

    return html


def update_version_history(*args):
    """Wrapper to update version history, ignoring any input args from previous step."""
    return build_version_history_html()


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
    """Run self-critique with chain-of-thought analysis streaming."""
    global current_article_clean, current_grounding_context, article_history, original_article

    if not current_article_clean:
        error_msg = "⚠️ No article to critique. Please generate an article first."
        return original_article, error_msg, ""

    # Center panel: original article (before critique)
    center_display = "📄 ORIGINAL ARTICLE\n"
    center_display += "=" * 44 + "\n\n"
    center_display += original_article

    # Left panel: critique analysis (streaming)
    critique_analysis = "🔍 SELF-CRITIQUE ANALYSIS\n"
    critique_analysis += "=" * 44 + "\n\n"
    critique_analysis += "⏳ Analyzing article for epistemic issues...\n\n"

    yield center_display, critique_analysis, ""

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
                critique_analysis_display = "🔍 SELF-CRITIQUE ANALYSIS\n"
                critique_analysis_display += "=" * 44 + "\n\n"
                critique_analysis_display += full_critique
                yield center_display, critique_analysis_display, ""

        # Step 2: Generate refined article based on critique
        critique_analysis_display = "🔍 SELF-CRITIQUE ANALYSIS\n"
        critique_analysis_display += "=" * 44 + "\n\n"
        critique_analysis_display += full_critique + "\n\n"
        critique_analysis_display += "✓ Analysis complete. Generating refined article...\n"
        yield center_display, critique_analysis_display, ""

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

        # Right panel: refined article (streaming)
        right_display = "📝 REFINED ARTICLE\n"
        right_display += "=" * 44 + "\n\n"
        right_display += "⏳ Generating refined version...\n\n"
        yield center_display, critique_analysis_display, right_display

        refined_article = ""
        for response, chunk in refinement_chat.stream():
            if chunk.content:
                refined_article += chunk.content
                refined_display = "📝 REFINED ARTICLE\n"
                refined_display += "=" * 44 + "\n\n"
                refined_display += refined_article
                yield center_display, critique_analysis_display, refined_display

        # Update current article for iterative critiques
        current_article_clean = refined_article

        # Store refined article in history
        final_article = f"📝 REFINED ARTICLE (Post-Critique)\n"
        final_article += "=" * 44 + "\n\n"
        final_article += f"__Epistemically refined through self-critique__\n\n---\n\n{refined_article}"
        article_history.append(final_article)

        # Final yield with complete versions
        final_critique_display = "🔍 SELF-CRITIQUE ANALYSIS\n"
        final_critique_display += "=" * 44 + "\n\n"
        final_critique_display += full_critique + "\n\n"
        final_critique_display += "✅ Critique complete! Refined article generated."

        yield center_display, final_critique_display, refined_display

    except Exception as e:
        error_analysis = critique_analysis + \
            f"\n\n❌ Error during critique: {str(e)}\n\nPlease try again."
        yield center_display, error_analysis, ""


def run_user_feedback(user_feedback):
    """Process user feedback on current article with validation and critical evaluation."""
    global current_article_clean, current_grounding_context, article_history, original_article

    if not current_article_clean:
        error_msg = "⚠️ No article available. Please generate an article first."
        center_display = "📝 CURRENT ARTICLE\n" + "=" * 44 + "\n\n" + \
            "No article available. Generate an article before providing feedback."
        rejection_display = "❌ FEEDBACK REJECTED\n" + "=" * 44 + "\n\n" + \
            "No article available to provide feedback on.\n\nGenerate an article first using Article Generation."
        return center_display, error_msg, rejection_display

    # Store original article for comparison
    original_article_text = current_article_clean

    # Center panel: current article (read-only reference)
    center_display = "📝 CURRENT ARTICLE\n"
    center_display += "=" * 44 + "\n\n"
    center_display += current_article_clean

    # Left panel: show processing message temporarily
    processing_msg = "⏳ Processing your feedback...\n\nValidating suggestions against source material..."

    # Right panel: placeholder during processing
    right_placeholder = "📋 CHANGELOG\n" + "=" * 44 + "\n\n⏳ Processing...\n\nChangelog will appear here."

    yield center_display, processing_msg, right_placeholder

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
            yield center_display, user_feedback, rejection_display
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
                yield streaming_center, processing_msg, right_placeholder

        # Step 4: Generate changelog showing what changed
        changelog_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        changelog_chat.append(system(
            "You are generating a changelog that shows what changed in the article based on user feedback.\n\n"
            "Create a clear, structured changelog with these sections:\n"
            "✓ ACCEPTED - Suggestions that were fully incorporated\n"
            "⚠️ PARTIALLY ACCEPTED - Suggestions that were modified before incorporating\n"
            "✗ REJECTED - Suggestions that were not incorporated\n\n"
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
                yield final_center, processing_msg, streaming_changelog

        # Update current article
        current_article_clean = revised_article

        # Store in history
        final_article = f"📝 REVISED ARTICLE (User Feedback)\n"
        final_article += "=" * 44 + "\n\n"
        final_article += f"__Revised based on user feedback__\n\n---\n\n{revised_article}"
        article_history.append(final_article)

        # Final display - use placeholder for "ready for next round" message
        final_center = revision_header + revised_article
        final_changelog = changelog_header + changelog

        # Return with empty value and placeholder for next round
        # Need to return a dict with both value and placeholder
        yield final_center, gr.update(value="", placeholder="✅ Feedback processed! Article updated.\n\nEnter new feedback to continue refining..."), final_changelog

    except Exception as e:
        error_display = f"❌ Error processing feedback: {str(e)}\n\nPlease try again."
        yield center_display, user_feedback, error_display


def run_synthetic_data_generation(topic, num_examples, quality_dist, flaw_type, article_length):
    """Generate synthetic training data with controlled epistemic properties."""
    import json
    import datetime

    # Convert num_examples from string to int
    num_examples = int(num_examples)

    # Map article length to word count range
    if article_length == "Brief":
        word_range = "75-100"
    elif article_length == "Standard":
        word_range = "175-200"
    elif article_length == "Long":
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
    log += f"- Quality distribution: Varied\n"
    log += f"- Flaws: Controlled injection\n\n"
    log += "⏳ Starting generation process...\n\n"

    yield "", log, ""

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

            # Map quality to specific epistemic flaws
            flaw_map = {
                "excellent": "none",
                "good": "minor_overcertainty",
                "fair": "missing_citations",
                "poor": "multiple_flaws",
                "terrible": "severe_bias_and_overstatements"
            }
            target_flaw = flaw_map[quality]

            # Update log
            log += f"📝 Generating example {i+1}/{num_examples}...\n"
            log += f"   Quality tier: {quality}\n"
            log += f"   Target flaw: {target_flaw}\n\n"
            yield center_preview, log, metadata_display

            # Generate article with controlled epistemic quality
            generator_chat = client.chat.create(model="grok-4-1-fast-reasoning")

            # Craft prompt based on target quality
            if quality == "excellent":
                quality_instruction = """Generate an epistemically excellent article with:
- Appropriate hedging and uncertainty language
- Well-sourced claims with clear citations
- Balanced framing of controversial points
- Clear distinction between evidence and speculation
- No overstatements or unwarranted certainty"""
            elif quality == "good":
                quality_instruction = """Generate a generally good article but with minor epistemic issues:
- Mostly appropriate certainty language, but 1-2 slightly overconfident claims
- Good sourcing overall
- Balanced framing
- One or two statements could use more qualification"""
            elif quality == "fair":
                quality_instruction = """Generate an article with noticeable epistemic issues:
- Several unsourced claims that should have citations
- Some appropriate hedging, but several definitive statements on uncertain matters
- Generally balanced but missing some important caveats
- Mix of good and problematic epistemic practices"""
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

            generator_chat.append(system(
                f"You are generating a synthetic training example for an epistemic quality classifier.\n\n"
                f"{quality_instruction}\n\n"
                f"Write a {word_range} word article about the given topic with the specified epistemic characteristics. "
                f"Output ONLY the article text in markdown format. Include a brief Sources section if appropriate."
            ))
            generator_chat.append(user(f"Generate an article about: {topic}"))

            # Generate article
            generated_article = ""
            for response, chunk in generator_chat.stream():
                if chunk.content:
                    generated_article += chunk.content

            # Label the article with epistemic scores
            labeler_chat = client.chat.create(model="grok-4-1-fast-reasoning")
            labeler_chat.append(system(
                "You are an epistemic quality labeler. Analyze the article and provide scores (0-10) for:\n"
                "- source_quality: How well claims are sourced and cited\n"
                "- certainty_appropriateness: Whether certainty language matches evidence strength\n"
                "- bias_level: How balanced vs biased the framing is (10 = very balanced, 0 = very biased)\n"
                "- completeness: Whether important caveats and limitations are mentioned\n\n"
                "Also identify specific epistemic flaws present.\n\n"
                "Format your response as:\n"
                "SOURCE_QUALITY: [0-10]\n"
                "CERTAINTY: [0-10]\n"
                "BIAS: [0-10]\n"
                "COMPLETENESS: [0-10]\n"
                "FLAWS: [comma-separated list of specific issues]"
            ))
            labeler_chat.append(user(f"Label this article:\n\n{generated_article}"))

            # Get labels
            label_response = ""
            for response, chunk in labeler_chat.stream():
                if chunk.content:
                    label_response += chunk.content

            # Parse labels
            import re
            source_quality = int(re.search(r'SOURCE_QUALITY:\s*(\d+)', label_response).group(1)) if re.search(r'SOURCE_QUALITY:\s*(\d+)', label_response) else 5
            certainty = int(re.search(r'CERTAINTY:\s*(\d+)', label_response).group(1)) if re.search(r'CERTAINTY:\s*(\d+)', label_response) else 5
            bias = int(re.search(r'BIAS:\s*(\d+)', label_response).group(1)) if re.search(r'BIAS:\s*(\d+)', label_response) else 5
            completeness = int(re.search(r'COMPLETENESS:\s*(\d+)', label_response).group(1)) if re.search(r'COMPLETENESS:\s*(\d+)', label_response) else 5
            flaws_match = re.search(r'FLAWS:\s*(.+)', label_response)
            flaws = flaws_match.group(1).strip() if flaws_match else "none identified"

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

            # Update center preview with latest article
            center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"
            center_preview += f"**Example {i+1}/{num_examples}** (Quality: {quality})\n"
            center_preview += "-" * 44 + "\n\n"
            center_preview += generated_article + "\n\n"
            if i > 0:
                center_preview += f"\n\n... ({i} previous examples generated)\n"

            # Update metadata display
            metadata_display = "📋 DATASET METADATA\n" + "=" * 44 + "\n\n"
            for entry in synthetic_dataset:
                metadata_display += f"**Example {entry['id']}** - {entry['target_quality'].upper()}\n"
                metadata_display += f"Scores: Src={entry['labels']['source_quality']}/10, "
                metadata_display += f"Cert={entry['labels']['certainty_appropriateness']}/10, "
                metadata_display += f"Bias={entry['labels']['bias_balance']}/10, "
                metadata_display += f"Comp={entry['labels']['completeness']}/10\n"
                metadata_display += f"Flaws: {entry['identified_flaws'][:80]}...\n\n"

            # Update log
            log += f"✅ Example {i+1} complete\n"
            log += f"   Scores: {source_quality}, {certainty}, {bias}, {completeness}\n\n"

            yield center_preview, log, metadata_display

        # Export to JSONL
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_data_{topic.replace(' ', '_')}_{timestamp}.jsonl"

        with open(filename, 'w', encoding='utf-8') as f:
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
            count = sum(1 for e in synthetic_dataset if e['target_quality'] == tier)
            metadata_display += f"  - {tier}: {count}\n"
        metadata_display += f"\n📁 Saved to: {filename}\n"

        yield center_preview, log, metadata_display

    except Exception as e:
        error_log = log + f"\n\n❌ Error during generation: {str(e)}\n\nPlease try again."
        yield center_preview, error_log, metadata_display


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
            "You are the Defender agent in an epistemic debate. Your role is to identify the strongest "
            "epistemic qualities of the article: appropriate certainty language, balanced framing, "
            "acknowledgment of limitations, and clear communication. Argue for what the article does well "
            "in terms of epistemic integrity. Be rigorous but fair. Provide a concise defense (~200 words)."
        ))
        defender_chat.append(user(
            f"Defend the epistemic strengths of this article:\n\n{current_article_clean}"))

        # Stream Defender response
        defender_response = ""
        debate_base = "🎭 MULTI-AGENT DEBATE\n" + "=" * 44 + \
            "\n\n⏳ Initializing debate agents...\n\n🟢 DEFENDER AGENT\n" + "-" * 44 + "\n"

        for response, chunk in defender_chat.stream():
            if chunk.content:
                defender_response += chunk.content
                streaming_transcript = debate_base + defender_response
                yield left_display, streaming_transcript, ""

        debate_transcript = debate_base + f"{defender_response}\n\n\n"
        yield left_display, debate_transcript, ""

        # Agent 2: Challenger
        debate_transcript += "🔴 CHALLENGER AGENT\n"
        debate_transcript += "-" * 44 + "\n"
        debate_transcript += "⏳ Analyzing article for weaknesses...\n\n"
        yield left_display, debate_transcript, ""

        challenger_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        challenger_chat.append(system(
            "You are the Challenger agent in an epistemic debate. Your role is to critically examine "
            "the article for epistemic weaknesses: overstatements, unwarranted certainty, missing caveats, "
            "one-sided framing, lack of epistemic humility, or unclear communication. Focus on how claims "
            "are presented, not their factual accuracy. Be aggressive but fair. Point out specific issues. "
            "Provide a concise critique (~200 words)."
        ))
        challenger_chat.append(user(
            f"Challenge the epistemic quality of this article:\n\n{current_article_clean}"))

        # Stream Challenger response
        challenger_response = ""
        challenger_base = debate_transcript + "🔴 CHALLENGER AGENT\n" + "-" * 44 + "\n"

        for response, chunk in challenger_chat.stream():
            if chunk.content:
                challenger_response += chunk.content
                streaming_transcript = challenger_base + challenger_response
                yield left_display, streaming_transcript, ""

        debate_transcript = challenger_base + f"{challenger_response}\n\n\n"
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

Produce the final revised article."""))

        # Stream Arbiter's revised article to RIGHT panel
        revised_article = ""
        arbiter_header = "📝 REVISED ARTICLE\n" + "=" * 44 + "\n\n"

        for response, chunk in arbiter_chat.stream():
            if chunk.content:
                revised_article += chunk.content
                streaming_right = arbiter_header + revised_article
                yield left_display, debate_transcript, streaming_right

        # Add completion message to debate transcript
        debate_transcript += "⚖️ ARBITER AGENT\n"
        debate_transcript += "-" * 44 + "\n"
        debate_transcript += "✅ Debate complete! Revised article generated.\n\n"
        debate_transcript += "**Key improvements:**\n"
        debate_transcript += "- Incorporated Defender's supporting evidence\n"
        debate_transcript += "- Addressed Challenger's valid criticisms\n"
        debate_transcript += "- Added epistemic qualifiers where appropriate\n"

        # Right panel: revised article (already streamed, just format final version)
        right_display = arbiter_header + revised_article

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
        search_url = "https://export.arxiv.org/api/query"

        params = {
            "search_query": f"all:{topic}",
            "start": 0,
            "max_results": 1,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        headers = {
            "User-Agent": "Veritas-Epistemics/1.0 (Educational Article Generator; mailto:user@example.com)"
        }
        response = requests.get(search_url, params=params,
                                headers=headers, timeout=10)

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

        # Stream the article generation
        article_content = ""

        # Build sources display for right panel during streaming
        streaming_sources_display = ""
        if len(current_sources) == 2:
            streaming_sources_display = "📚 WIKIPEDIA & ARXIV ARTICLES\n"
            streaming_sources_display += "=" * 44 + "\n\n"
            streaming_sources_display += "WIKIPEDIA:\n"
            streaming_sources_display += "-" * 44 + "\n\n"
            streaming_sources_display += f"**{current_sources[0]['name']}**\n\n"
            streaming_sources_display += f"Source: {current_sources[0]['url']}\n\n"
            streaming_sources_display += f"{current_sources[0]['content'][:2500]}\n\n\n"
            streaming_sources_display += "ARXIV:\n"
            streaming_sources_display += "-" * 44 + "\n\n"
            streaming_sources_display += f"**{current_sources[1]['name']}**\n\n"
            streaming_sources_display += f"Authors: {current_sources[1]['authors']}\n\n"
            streaming_sources_display += f"Source: {current_sources[1]['url']}\n\n"
            streaming_sources_display += f"{current_sources[1]['content'][:2500]}"
        elif len(current_sources) == 1:
            source = current_sources[0]
            if source['type'] == 'Wikipedia':
                streaming_sources_display = "📚 WIKIPEDIA ARTICLE\n"
                streaming_sources_display += "=" * 44 + "\n\n"
                streaming_sources_display += f"**{source['name']}**\n\n"
                streaming_sources_display += f"Source: {source['url']}\n\n"
                streaming_sources_display += "---\n\n"
                streaming_sources_display += f"{source['content'][:2500]}"
            else:
                streaming_sources_display = "📚 ARXIV ARTICLE\n"
                streaming_sources_display += "=" * 44 + "\n\n"
                streaming_sources_display += f"**{source['name']}**\n\n"
                streaming_sources_display += f"Authors: {source['authors']}\n\n"
                streaming_sources_display += f"Source: {source['url']}\n\n"
                streaming_sources_display += "---\n\n"
                streaming_sources_display += f"{source['content'][:2500]}"

        for response, chunk in chat.stream():
            if chunk.content:
                article_content += chunk.content

                # Build streaming article display
                streaming_article = "📝 YOUR ARTICLE\n"
                streaming_article += "=" * 44 + "\n\n"
                streaming_article += f"__{source_note}__\n\n---\n\n{article_content}"

                # Yield: center (streaming article), left (status log), right (sources)
                yield streaming_article, status_log, streaming_sources_display

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

        # Yield: center (article), left (status), right (sources) - reuse streaming_sources_display
        yield final_article, status_log, streaming_sources_display

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
            background-color: #0F0F0F !important;
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
            right: -450px;
            width: 400px;
            height: 100vh;
            background-color: #0f0f0f;
            border-left: 3px solid #6366f1;
            z-index: 9999;
            overflow-y: auto;
            padding: 20px;
            box-shadow: -4px 0 20px rgba(0, 0, 0, 0.5);
            transition: right 0.3s ease;
            pointer-events: none;
        }

        #version-panel.visible {
            right: 0;
            pointer-events: auto;
        }

        #close-version-panel:hover {
            color: #f87171;
            transform: scale(1.2);
        }

        .version-item {
            background-color: #1a1a1a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .version-item:hover {
            border-color: #6366f1;
            background-color: #252525;
        }

        .version-item.latest {
            border-color: #10b981;
            background-color: #1a2420;
        }

        .version-label {
            font-weight: bold;
            color: #e5e7eb;
            margin-bottom: 4px;
        }

        .version-type {
            font-size: 0.85rem;
            color: #9ca3af;
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
            padding: 12px !important;
            margin-bottom: 10px !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }

        .version-item:hover {
            border-color: #6366f1 !important;
            background-color: #252525 !important;
        }

        .version-item.latest {
            border-color: #10b981 !important;
            background-color: #1a2420 !important;
        }

        .version-label {
            font-weight: bold !important;
            color: #e5e7eb !important;
            margin-bottom: 4px !important;
        }

        .version-type {
            font-size: 0.85rem !important;
            color: #9ca3af !important;
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
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 1.2rem; font-weight: bold; color: #6366f1; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #333;">
                        <span>📜 VERSION HISTORY</span>
                        <button id="close-version-panel" onclick="document.getElementById('version-panel').classList.remove('visible')" style="background: none; border: none; color: #6366f1; font-size: 1.5rem; cursor: pointer; padding: 0; line-height: 1;">&times;</button>
                    </div>
                    <div id="version-list">
                        <div style='color: #9ca3af; text-align: center; padding: 20px;'>No versions yet. Generate an article to begin.</div>
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

        // Helper to update version list from Gradio state
        function updateVersionList(html) {
            const versionList = document.getElementById('version-list');
            if (versionList && html) {
                versionList.innerHTML = html;
            }
        }

        // Function called when clicking a version item
        function selectVersion(index) {
            // TODO: Implement loading specific version into center panel
            // For now, just close the panel
            const panel = document.getElementById('version-panel');
            if (panel) {
                panel.classList.remove('visible');
            }
        }
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
                choices=["Brief", "Standard", "Long"],
                value="Standard",
                label="Article Length",
                info="Target word count range",
                elem_id="length-dropdown"
            )

        # Hidden placeholder for synthetic_log (removed from display but needed for outputs)
        synthetic_log = gr.Textbox(visible=False)

        # Normal left panel - Process log and status (visible for all tools EXCEPT Synthetic Data)
        left_panel = gr.Textbox(
            value="🔍 PROCESS LOG\n" + "=" * 42 + "\n\nThis panel displays real-time status updates during article generation:\n\n- Wikipedia search progress\n- arXiv paper search progress\n- Source retrieval status\n- Article generation steps\n- Completion notifications\n\nEnter a topic and click the arrow to begin!",
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
            "\n\nYour generated article will appear here.\n\nIterate it in order to get as close to the truth as you can!",
            lines=30,
            interactive=False,
            show_copy_button=False,
            container=True,
            elem_classes=["side-panel", "central-article"],
            show_label=False
        )

        # Right panel - Source material (Wikipedia & arXiv)
        right_panel = gr.Textbox(
            value="📚 SOURCE MATERIAL\n" + "=" * 42 +
            "\n\nThis panel displays source articles used to generate your article:\n\n- Wikipedia articles (general knowledge)\n- arXiv papers (academic research)\n- Source URLs and titles\n- Reference material for verification\n\nSources will appear here after article generation.",
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

    # Route action button to correct function based on dropdown
    def execute_action(selected_tool, topic, user_feedback, num_examples, quality_dist, flaw_type, article_length):
        if selected_tool == "Article Generation":
            # Generate article - yields to (center, left, right)
            for center, left, right in generate_initial_article(topic):
                yield center, left, left, right  # Duplicate left for both panels
        elif selected_tool == "Multi-Agent Debate":
            # Run debate
            for center, left, right in run_multi_agent_debate():
                yield center, left, left, right
        elif selected_tool == "Self-Critique":
            # Run self-critique
            for center, left, right in run_self_critique():
                yield center, left, left, right
        elif selected_tool == "User Feedback":
            # Process user feedback
            for center, left, right in run_user_feedback(user_feedback):
                yield center, left, left, right
        elif selected_tool == "Synthetic Data":
            # Generate synthetic training data - yields to (center, synthetic_log, right)
            for center, synth_log, right in run_synthetic_data_generation(topic, num_examples, quality_dist, flaw_type, article_length):
                yield center, "", synth_log, right  # Empty string for left_panel, actual log for synthetic_log
        else:
            # Placeholder for other tools
            error_msg = f"⚠️ {selected_tool} not yet implemented."
            yield "", error_msg, error_msg, ""

    # Trigger action on Enter key in topic input (same behavior as action button)
    topic_input.submit(
        fn=execute_action,
        inputs=[epistemic_dropdown, topic_input, left_panel, num_examples_number, quality_dropdown, flaw_dropdown, length_dropdown],
        outputs=[article_display, left_panel, synthetic_log, right_panel]
    ).then(
        fn=update_version_history,
        inputs=[article_display, left_panel, right_panel],
        outputs=[version_state]
    ).then(
        fn=None,
        inputs=[version_state],
        js="""(versionHtml) => {
            setTimeout(() => {
                const textareas = document.querySelectorAll('textarea');
                textareas.forEach(t => { t.scrollTop = 0; });
            }, 100);

            const versionList = document.getElementById('version-list');
            if (versionList && versionHtml) {
                versionList.innerHTML = versionHtml;
            }
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

        # Disable topic input when epistemic tool is selected (except Synthetic Data which allows any topic)
        topic_disabled = (selected_tool not in ["Article Generation", "Synthetic Data"])

        # Check if article exists for epistemic tools (Synthetic Data doesn't need an existing article)
        button_disabled = topic_disabled and not current_article_clean

        # Set placeholder content for panels based on selected tool
        if selected_tool == "Article Generation":
            left_placeholder = "🔍 PROCESS LOG\n" + "=" * 42 + "\n\nThis panel displays real-time status updates during article generation:\n\n- Wikipedia search progress\n- arXiv paper search progress\n- Source retrieval status\n- Article generation steps\n- Completion notifications\n\nEnter a topic and click the button to begin!"
            center_placeholder = "📝 YOUR ARTICLE\n" + "=" * 42 + \
                "\n\nYour generated article will appear here.\n\nIterate it in order to get as close to the truth as you can!"
            right_placeholder = "📚 SOURCE MATERIAL\n" + "=" * 42 + \
                "\n\nThis panel displays source articles used to generate your article:\n\n- Wikipedia articles (general knowledge)\n- arXiv papers (academic research)\n- Source URLs and titles\n- Reference material for verification\n\nSources will appear here after article generation."

        elif selected_tool == "Self-Critique":
            left_placeholder = "🔍 CRITIQUE ANALYSIS\n" + "=" * 42 + "\n\nThis panel displays the critical analysis of your article:\n\n- Epistemic quality assessment\n- Identification of overstatements\n- Analysis of certainty language\n- Detection of missing qualifiers\n- Suggestions for improvement\n\nClick 'Critique Article' to begin the self-critique process!"

            # If article exists, show it with the "ORIGINAL ARTICLE" header
            if current_article_clean:
                center_placeholder = "📝 ORIGINAL ARTICLE\n" + "=" * 42 + "\n\n" + current_article_clean
            else:
                center_placeholder = "📝 ORIGINAL ARTICLE\n" + "=" * 42 + \
                    "\n\nYour current article will be displayed here during the critique process.\n\nThe AI will analyze its epistemic quality and suggest improvements."

            right_placeholder = "✨ REFINED ARTICLE\n" + "=" * 42 + "\n\nThis panel displays the epistemically refined version of your article:\n\n- Improved certainty language\n- Added qualifiers where needed\n- Balanced framing\n- Clearer communication\n- Enhanced epistemic integrity\n\nThe refined article will appear here after critique completes."

        elif selected_tool == "Multi-Agent Debate":
            left_placeholder = "🎭 DEBATE TRANSCRIPT\n" + "=" * 42 + "\n\nThis panel displays the multi-agent debate transcript:\n\n- Defender: Argues for epistemic strengths\n- Challenger: Identifies epistemic weaknesses\n- Arbiter: Synthesizes debate into improvements\n\nThe debate process analyzes the article from multiple perspectives to produce a more balanced revision.\n\nClick 'Start Debate' to begin the multi-agent debate!"

            # If article exists, show it with the "ORIGINAL ARTICLE" header
            if current_article_clean:
                center_placeholder = "📝 ORIGINAL ARTICLE\n" + "=" * 42 + "\n\n" + current_article_clean
            else:
                center_placeholder = "📝 ORIGINAL ARTICLE\n" + "=" * 42 + \
                    "\n\nYour current article will be displayed here during the debate.\n\nThree agents will analyze its epistemic quality through structured debate."

            right_placeholder = "📝 REVISED ARTICLE\n" + "=" * 42 + "\n\nThis panel displays the revised article produced by the Arbiter:\n\n- Incorporates Defender's supporting evidence\n- Addresses Challenger's valid criticisms\n- Adds epistemic qualifiers where appropriate\n- Balances competing perspectives\n\nThe revised article will appear here after debate completes."

        elif selected_tool == "User Feedback":
            # Left panel becomes interactive for user input - use placeholder instead of value
            left_panel_placeholder = "Enter your feedback here...\n\nSuggest improvements to:\n- Claims needing qualifiers or sources\n- Overstatements or unwarranted certainty\n- Missing perspectives or caveats\n- Structural or clarity issues\n\nBe specific and reference particular sections."
            left_placeholder = ""  # Empty value, will use placeholder instead

            # If article exists, show it in center
            if current_article_clean:
                center_placeholder = "📝 CURRENT ARTICLE\n" + "=" * 42 + "\n\n" + current_article_clean
            else:
                center_placeholder = "📝 CURRENT ARTICLE\n" + "=" * 42 + \
                    "\n\nNo article available. Generate an article before providing feedback."

            right_placeholder = "📋 CHANGELOG\n" + "=" * 42 + "\n\nThis panel will display the changelog after processing your feedback:\n\n✓ ACCEPTED changes\n   - What was incorporated and why\n\n⚠️ PARTIALLY ACCEPTED changes\n   - What was modified and reasoning\n\n✗ REJECTED changes\n   - Why suggestions weren't incorporated\n\nThe revised article will appear in the center panel.\n\nEnter your feedback in the left panel and click 'Collect Feedback'."

        elif selected_tool == "Synthetic Data":
            # Synthetic Data Generation placeholders
            left_placeholder = "🏭 GENERATION LOG\n" + "=" * 42 + "\n\nThis panel displays the synthetic data generation process:\n\n- Configuration details\n- Generation progress for each example\n- Quality tier assignments\n- Epistemic flaw injection\n- Export status and file location\n\nEnter a topic and click 'Generate Data' to create labeled training examples!"

            center_placeholder = "📝 ARTICLE PREVIEW\n" + "=" * 42 + "\n\nThis panel will display generated articles as they are created:\n\n- Each article with its target quality level\n- Controlled epistemic characteristics\n- Varied certainty language patterns\n- Different sourcing qualities\n\nGenerated articles will appear here during the process."

            right_placeholder = "📋 METADATA & LABELS\n" + "=" * 42 + "\n\nThis panel displays structured metadata for each generated example:\n\n- Epistemic quality scores (0-10)\n- Identified flaws and issues\n- Target quality tier\n- Dataset summary statistics\n- Export file information\n\nLabeled data suitable for training classifiers will appear here."

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

        # For User Feedback, clear value and use placeholder; for others, use value
        if selected_tool == "User Feedback":
            return (
                gr.update(value=button_text, interactive=not button_disabled),
                gr.update(interactive=not topic_disabled),
                gr.update(value=center_placeholder),
                gr.update(value="", placeholder=left_panel_placeholder, interactive=left_interactive, visible=show_normal_left),
                gr.update(value=right_placeholder),
                gr.update(visible=show_synthetic_panel)  # synthetic_left_panel
            )
        else:
            return (
                gr.update(value=button_text, interactive=not button_disabled),
                gr.update(interactive=not topic_disabled),
                gr.update(value=center_placeholder),
                gr.update(value=left_placeholder, interactive=left_interactive, visible=show_normal_left),
                gr.update(value=right_placeholder),
                gr.update(visible=show_synthetic_panel)  # synthetic_left_panel
            )

    epistemic_dropdown.change(
        fn=update_ui_state,
        inputs=[epistemic_dropdown],
        outputs=[action_button, topic_input,
                 article_display, left_panel, right_panel,
                 synthetic_left_panel]
    )

    action_button.click(
        fn=execute_action,
        inputs=[epistemic_dropdown, topic_input, left_panel, num_examples_number, quality_dropdown, flaw_dropdown, length_dropdown],
        outputs=[article_display, left_panel, synthetic_log, right_panel]
    ).then(
        fn=update_version_history,
        inputs=[article_display, left_panel, right_panel],
        outputs=[version_state]
    ).then(
        fn=None,
        inputs=[version_state],
        js="""(versionHtml) => {
            setTimeout(() => {
                const textareas = document.querySelectorAll('textarea');
                textareas.forEach(t => { t.scrollTop = 0; });
            }, 100);

            const versionList = document.getElementById('version-list');
            if (versionList && versionHtml) {
                versionList.innerHTML = versionHtml;
            }
        }"""
    )

    # Version History button click handler
    version_history_btn.click(
        fn=toggle_version_panel,
        outputs=[version_state]
    ).then(
        fn=None,
        inputs=[version_state],
        js="""(versionListHtml) => {
            // Ensure panel exists
            if (!document.getElementById('version-panel')) {
                const panel = document.createElement('div');
                panel.id = 'version-panel';
                panel.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 1.2rem; font-weight: bold; color: #6366f1; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #333;">
                        <span>📜 VERSION HISTORY</span>
                        <button id="close-version-panel" onclick="document.getElementById('version-panel').classList.remove('visible')" style="background: none; border: none; color: #6366f1; font-size: 1.5rem; cursor: pointer; padding: 0; line-height: 1;">&times;</button>
                    </div>
                    <div id="version-list">
                        <div style='color: #9ca3af; text-align: center; padding: 20px;'>No versions yet. Generate an article to begin.</div>
                    </div>
                `;
                document.body.appendChild(panel);
            }

            // Update the version list content
            const versionList = document.getElementById('version-list');
            if (versionList && versionListHtml) {
                versionList.innerHTML = versionListHtml;
            }

            // Toggle panel visibility
            const panel = document.getElementById('version-panel');
            if (panel) {
                panel.classList.toggle('visible');
            }
        }"""
    )


# Launch without footer
demo.launch(show_api=False)
