# feedback.py
# User feedback processing functionality

import os
from datetime import datetime
import gradio as gr

from config import client
from xai_sdk.chat import user, system


def run_user_feedback(user_feedback: str, current_article_clean: str, current_grounding_context: str):
    """Process user feedback on current article with validation and critical evaluation.

    Args:
        user_feedback: The user's feedback text
        current_article_clean: The current article content (without headers)
        current_grounding_context: The source context for fact-checking

    Yields: (center, left, right, filepath, state_updates)

    state_updates is None during streaming, and on final yield contains:
    {
        'current_article_clean': str,  # Updated article content
        'article_history_entry': str   # Entry to append to article_history
    }
    """
    if not current_article_clean:
        error_msg = "⚠️ No article available. Please generate an article first."
        center_display = "📝 CURRENT ARTICLE\n" + "=" * 44 + "\n\n" + \
            "No article available. Generate an article before providing feedback."
        rejection_display = "❌ FEEDBACK REJECTED\n" + "=" * 44 + "\n\n" + \
            "No article available to provide feedback on.\n\nGenerate an article first using Article Generation."
        return center_display, error_msg, rejection_display, None, None

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

    yield center_display, processing_msg, right_placeholder, None, None

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
            yield center_display, user_feedback, rejection_display, None, None
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
                yield streaming_center, processing_msg, right_placeholder, None, None

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
                yield final_center, processing_msg, streaming_changelog, None, None

        # Build the article history entry
        final_article = f"📝 REVISED ARTICLE (User Feedback)\n"
        final_article += "=" * 44 + "\n\n"
        final_article += f"__Revised based on user feedback__\n\n---\n\n{revised_article}"

        # Save revised article to file for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_{timestamp}.md"
        filepath = os.path.abspath(filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(revised_article)
        except Exception as e:
            print(f"Error saving feedback revised article: {e}")
            filepath = None

        # Build state updates for main.py to apply
        state_updates = {
            'current_article_clean': revised_article,
            'article_history_entry': final_article
        }

        # Final display - use placeholder for "ready for next round" message
        final_center = revision_header + revised_article
        final_changelog = changelog_header + changelog

        # Return with gr.update for the left panel to clear it and set placeholder
        yield final_center, gr.update(value="", placeholder="✅ Feedback processed! Article updated.\n\nEnter new feedback to continue refining..."), final_changelog, filepath, state_updates

    except Exception as e:
        error_display = f"❌ Error processing feedback: {str(e)}\n\nPlease try again."
        yield center_display, user_feedback, error_display, None, None
