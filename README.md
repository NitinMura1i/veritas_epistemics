# Veritas Epistemics

An epistemic refinement tool for AI-generated content, powered by Grok.

Generate articles from topics or URLs, then refine them through multi-agent debate, self-critique, and validated user feedback to improve epistemic quality.

<img width="1920" height="1080" alt="front" src="https://github.com/user-attachments/assets/7538e9ef-eacb-4fd6-a5c6-e2b0e657a778" />

## Features

- **Article Generation** - Generate grounded articles from any topic or URL using Grok
- **Multi-Agent Debate** - Defender and Challenger agents argue over the article's epistemic merits, with an Arbiter producing a balanced revision
- **Self-Critique** - Automated analysis identifying overstatements, missing caveats, and unsupported claims
- **User Feedback** - Submit your own suggestions, which are validated and selectively incorporated
- **Synthetic Data Generation** - Generate training data for epistemic fine-tuning
- **Version History** - Track all revisions and restore previous versions

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Veritas_Epistemics.git
cd Veritas_Epistemics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your xAI API key:
```
XAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python main.py
```

The app will open in your browser at `http://localhost:7860`

## Usage

1. Select a tool from the dropdown (Article Generation, Multi-Agent Debate, etc.)
2. Enter a topic or URL in the input field
3. Click the action button to process
4. Use the Version History panel to track and restore previous versions
5. Download any generated article using the download button

## Built With

- [Gradio](https://gradio.app/) - Web interface
- [Grok](https://x.ai/) - AI model via xAI API
- [Ray](https://ray.io/) - Parallel processing for synthetic data generation

## License

MIT License
