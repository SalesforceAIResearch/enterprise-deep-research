# Enterprise Deep Research

A comprehensive AI-powered research assistant with both Python backend and React frontend components.

## ✨ Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic, Google, Groq, and SambaNova
- **Advanced Web Search**: Powered by Tavily API for comprehensive research
- **Real-time Streaming**: Live research progress updates
- **File Analysis**: Support for PDF, TXT, and other document formats
- **Benchmark Mode**: Optimized for evaluation with full citation processing
- **React Frontend**: Modern, responsive user interface
- **Concurrent Processing**: Parallel query processing for faster evaluation

## 🛠️ Installation

### Prerequisites

- **Python 3.11+**
- **Node.js v20.9.0+**
- **npm package manager**

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/SalesforceAIResearch/enterprise-deep-research.git
   cd enterprise-deep-research
   ```

2. **Create and activate Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.sample .env
   ```
   
   Edit the `.env` file with your API keys and configuration.

## ⚙️ Configuration

### Environment Variables

#### Required
- `TAVILY_API_KEY` - Your Tavily search API key (always required)
- **One LLM provider key** (choose based on your preferred provider):
  - `OPENAI_API_KEY` - OpenAI API key
  - `ANTHROPIC_API_KEY` - Anthropic API key
  - `GROQ_API_KEY` - Groq API key
  - `GOOGLE_CLOUD_PROJECT` - Google Cloud project ID
  - `SAMBNOVA_API_KEY` - SambaNova API key

#### Optional
- `LLM_PROVIDER` - Default LLM provider (default: `openai`)
- `LLM_MODEL` - Default model name (provider-specific defaults)
- `MAX_WEB_RESEARCH_LOOPS` - Maximum research loops (default: `10`)
- `GOOGLE_CLOUD_LOCATION` - Google Cloud location (default: `us-central1`)

### Supported Models

| Provider | Environment Variable | Default Model | Available Models |
|----------|---------------------|---------------|------------------|
| **OpenAI** | `OPENAI_API_KEY` | `o4-mini` | `o4-mini`, `o4-mini-high`, `o3-mini`, `o3-mini-reasoning`, `gpt-4o` |
| **Anthropic** | `ANTHROPIC_API_KEY` | `claude-sonnet-4` | `claude-sonnet-4`, `claude-sonnet-4-thinking`, `claude-3-7-sonnet`, `claude-3-7-sonnet-thinking`, `claude-3-5-sonnet` |
| **Google** | `GOOGLE_CLOUD_PROJECT` | `gemini-2.5-pro` | `gemini-2.5-pro`, `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest`, `gemini-pro` |
| **Groq** | `GROQ_API_KEY` | `deepseek-r1-distill-llama-70b` | `deepseek-r1-distill-llama-70b`, `llama-3.3-70b-versatile`, `llama3-70b-8192` |
| **SambaNova** | `SAMBNOVA_API_KEY` | `DeepSeek-V3-0324` | `DeepSeek-V3-0324` |

5. **Install and build frontend**
   ```bash
   cd ai-research-assistant
   npm install && npm run build
   cd ..
   ```

6. **Start the application**
   
   **Option A: Full Stack (Frontend + Backend)**
   ```bash
   # Terminal 1 - Backend
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   
   # Terminal 2 - Frontend
   cd ai-research-assistant && npm start
   ```
   
   **Option B: Backend Only**
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

7. **Access the application**
   - Full Stack: http://localhost:3000
   - Backend Only: http://localhost:8000



## 🚀 Usage

### Basic Research Query
```bash
python benchmarks/run_research.py "Your research question here" \
  --provider openai \
  --model o3-mini \
  --max-loops 5
```

> Navigate to http://localhost:8000 to interact with the agent

## 📊 Benchmarking

For comprehensive benchmarking and evaluation capabilities, see our detailed [Benchmarking Guide](benchmarks/README.md).

Supported benchmarks:
- **DeepResearchBench** - Comprehensive research evaluation
- **ResearchQA** - Question-answering with citations
- **DeepConsult** - Consulting-style research tasks

## 📁 Project Structure

```
enterprise-deep-research/
├── ai-research-assistant/     # React frontend application
├── benchmarks/               # Evaluation scripts and datasets
├── src/                     # Core research engine
├── services/                # Backend services
├── app.py                   # Main FastAPI application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black src/ services/ benchmarks/
```

### Type Checking
```bash
mypy src/ services/
```

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md

## 📝 Citation

Coming Soon ...

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Powered by [Tavily](https://tavily.com) for web search capabilities
- Frontend built with [React](https://reactjs.org/) and [Tailwind CSS](https://tailwindcss.com/)
