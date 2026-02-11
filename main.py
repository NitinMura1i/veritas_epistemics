# main.py
# Gradio UI layout and event wiring

import gradio as gr

# Local imports
from styles import STYLES_AND_SCRIPTS
from synthetic import run_synthetic_data_generation
from article import generate_initial_article as _generate_initial_article
from critique import run_self_critique as _run_self_critique
from feedback import run_user_feedback as _run_user_feedback
from debate import run_multi_agent_debate as _run_multi_agent_debate

# Global state management (kept in main.py for proper 'global' keyword usage)
article_history = []
current_sources = []
source_visible = False  # Track if source panel is visible
current_article_clean = ""  # Store article content without headers for debate
current_grounding_context = ""  # Store grounding context for debates
original_article = ""  # Store original article before any debates
version_panel_visible = False  # Track if version history panel is visible


def generate_initial_article(topic: str):
    """Wrapper that calls article.py's generate_initial_article and handles global state updates."""
    global article_history, current_sources, current_article_clean, current_grounding_context, original_article

    # Clear version history for new topic
    article_history = []

    for center, left, right, filepath, state_updates in _generate_initial_article(topic):
        if state_updates is not None:
            # Final yield - apply state updates
            article_history.append(state_updates['article_history_entry'])
            current_sources = state_updates['current_sources']
            current_article_clean = state_updates['current_article_clean']
            current_grounding_context = state_updates['current_grounding_context']
            original_article = state_updates['original_article']
        yield center, left, right, filepath


def run_self_critique():
    """Wrapper that calls critique.py's run_self_critique and handles global state updates."""
    global current_article_clean, article_history

    for left, center, right, filepath, state_updates in _run_self_critique(current_article_clean, original_article):
        if state_updates is not None:
            # Final yield - apply state updates
            current_article_clean = state_updates['current_article_clean']
            article_history.append(state_updates['article_history_entry'])
        yield left, center, right, filepath


def run_user_feedback(user_feedback):
    """Wrapper that calls feedback.py's run_user_feedback and handles global state updates."""
    global current_article_clean, article_history

    for center, left, right, filepath, state_updates in _run_user_feedback(user_feedback, current_article_clean, current_grounding_context):
        if state_updates is not None:
            # Final yield - apply state updates
            current_article_clean = state_updates['current_article_clean']
            article_history.append(state_updates['article_history_entry'])
        yield center, left, right, filepath


def run_multi_agent_debate():
    """Wrapper that calls debate.py's run_multi_agent_debate and handles global state updates."""
    global current_article_clean, article_history

    for left, center, right, filepath, state_updates in _run_multi_agent_debate(current_article_clean, original_article):
        if state_updates is not None:
            # Final yield - apply state updates
            current_article_clean = state_updates['current_article_clean']
            article_history.append(state_updates['article_history_entry'])
        yield left, center, right, filepath


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
        # show_download_button=False,
        # show_fullscreen_button=False,
    )

    # CSS and JavaScript for UI styling
    gr.HTML(STYLES_AND_SCRIPTS)

    # Top control row: dropdown and input box
    with gr.Row(elem_classes=["top-control-row"]):
        # Epistemic tools dropdown
        epistemic_dropdown = gr.Dropdown(
            choices=[
                "Article Generation",
                "Multi-Agent Debate",
                "Self-Critique",
                "User Feedback",
                "Synthetic Data"
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
            # show_copy_button=False,
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
            # show_copy_button=False,
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
            # show_copy_button=False,
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

            // Store the latest version number for comparison
            window.latestVersionNum = (data.articles && data.articles.length > 0) ? data.articles[0].version : null;

            // Enable/disable restore button (need at least 2 versions, and start disabled since we view latest by default)
            const restoreBtn = document.getElementById('restore-version-btn');
            if (restoreBtn) {
                // Start disabled since we're viewing the latest version by default
                if (data.articles && data.articles.length > 1) {
                    // More than 1 version exists, but we're viewing latest so still disabled
                    restoreBtn.disabled = true;
                    restoreBtn.style.opacity = '0.5';
                    restoreBtn.style.cursor = 'not-allowed';
                    restoreBtn.style.color = '#555';
                } else {
                    // Only 1 version exists, definitely disabled
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
                    // Enable/disable restore button based on whether this is the latest version
                    const restoreBtn = document.getElementById('restore-version-btn');
                    if (restoreBtn) {
                        if (versionNum === window.latestVersionNum) {
                            // Viewing latest version - disable restore
                            restoreBtn.disabled = true;
                            restoreBtn.style.opacity = '0.5';
                            restoreBtn.style.cursor = 'not-allowed';
                            restoreBtn.style.color = '#555';
                        } else {
                            // Viewing older version - enable restore
                            restoreBtn.disabled = false;
                            restoreBtn.style.opacity = '1';
                            restoreBtn.style.cursor = 'pointer';
                            restoreBtn.style.color = '#fff';
                        }
                    }
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


# Launch with no footer (show_api=False) = no Gradio branding, and share=True to get a public link for 72 hours
demo.launch()
