# styles.py
# Contains all CSS and JavaScript for the Gradio UI

STYLES_AND_SCRIPTS = """
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
    """
