<p align="center">
  <img src="assets/edr-logo.png" alt="EDR Logo"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  </a>
  <a href="https://arxiv.org/abs/2510.17797">
    <img src="https://img.shields.io/badge/arXiv-2510.17797-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/Salesforce/EDR-200">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-EDR--200%20Dataset-blue" alt="HF Dataset">
  </a>
  <a href="https://deepwiki.com/SalesforceAIResearch/enterprise-deep-research">
    <img src="https://img.shields.io/badge/Documentation-DeepWiki-00A86B?style=flat&logo=book&logoColor=white" alt="DeepWiki Documentation">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>
  <a href="https://github.com/SalesforceAIResearch/enterprise-deep-research/stargazers">
    <img src="https://img.shields.io/github/stars/SalesforceAIResearch/enterprise-deep-research.svg" alt="GitHub stars">
  </a>
  <a href="https://x.com/SFResearch/status/1981831647277297799">
    <img src="https://img.shields.io/badge/Follow%20us%20on-X-1DA1F2?style=flat&logo=x&logoColor=white" alt="Follow us on X">
  </a>
  <a href="https://www.linkedin.com/posts/weiranyao_agenticai-deepresearch-enterpriseai-activity-7386248260388298752-Q6I4?utm_source=share&utm_medium=member_desktop&rcm=ACoAAB5LMXoBvpSjjHtk7HQatx3eDUEFvmo7azo">
    <img src="https://img.shields.io/badge/Follow%20us%20on-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="Follow us on LinkedIn">
  </a>
</p>


We present **Enterprise Deep Research (EDR)**, a multi-agent system that integrate: 
- Master Planning Agent for adaptive query decomposition.
- Four specialized search agents (General, Academic, GitHub, LinkedIn).
- Extensible MCP-based tool ecosystem supporting NL2SQL, file analysis, and enterprise workflows.
- Visualization Agent for data-driven insights. 
- Reflection mechanism that detects knowledge gaps and updates research direction with optional human-in-the-loop steering guidance. 
- Real-time steering commands for continuous research refinement.

> [!Note]
> These components enable automated report generation, real-time streaming, and seamless enterprise deployment, as validated on internal datasets.

![Architecture Overview](./assets/edr_ppl.png)

## 🎥 Demo

We present a video demo of using EDR in web application for enterprise deep data analysis. 

#### EDR: Web Application
<https://github.com/user-attachments/assets/b40b6131-3b7c-4e5f-8de3-91a4047ade58>

> [!Note]
> Multi-provider LLM support • Slack agent • Real-time streaming • Document analysis • Citation management • Parallel processing • Specialized benchmarking • Human-in-the-loop steering

## 🚀 Quick Start

**Requirements**: Python 3.11+ • Node.js 20.9.0+

### Installation & Setup

```bash
# Clone and setup
git clone https://github.com/SalesforceAIResearch/enterprise-deep-research.git
cd enterprise-deep-research

# Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.sample .env
# Edit .env with your API keys

# Frontend setup
cd ai-research-assistant && npm install && npm run build && cd ..
```

### Environment Configuration

**Required Variables:**
- `TAVILY_API_KEY` - Tavily search API key
- **One LLM provider key:**
  - `OPENAI_API_KEY` - OpenAI API key
  - `ANTHROPIC_API_KEY` - Anthropic API key  
  - `GROQ_API_KEY` - Groq API key
  - `GOOGLE_CLOUD_PROJECT` - Google Cloud project ID
  - `SAMBNOVA_API_KEY` - SambaNova API key

**Optional Settings:**
- `LLM_PROVIDER` - Default provider (default: `openai`)
- `LLM_MODEL` - Model name (provider-specific defaults)
- `MAX_WEB_RESEARCH_LOOPS` - Max iterations (default: `10`)

### Supported Models

| Provider | Default Model | Available Models |
|----------|---------------|------------------|
| **OpenAI** | `o4-mini` | `o4-mini`, `o4-mini-high`, `o3-mini`, `o3-mini-reasoning`, `gpt-4o` |
| **Anthropic** | `claude-sonnet-4` | `claude-sonnet-4`, `claude-sonnet-4-thinking`, `claude-3-7-sonnet`, `claude-3-7-sonnet-thinking` |
| **Google** | `gemini-2.5-pro` | `gemini-2.5-pro`, `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest` |
| **Groq** | `deepseek-r1-distill-llama-70b` | `deepseek-r1-distill-llama-70b`, `llama-3.3-70b-versatile`, `llama3-70b-8192` |
| **SambaNova** | `DeepSeek-V3-0324` | `DeepSeek-V3-0324` |

### Running the Application

**Full Stack (Recommended) - Single Command:**
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
The application will serve both the backend API and pre-built frontend at [http://localhost:8000](http://localhost:8000)

**Backend API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 💻 Usage

### Command Line
```bash
python benchmarks/run_research.py "Your research question" \
  --provider openai --model o3-mini --max-loops 3
```

### Web Interface
Navigate to [http://localhost:8000](http://localhost:8000) for interactive research with real-time progress tracking.

## 📚 Benchmarking & Development

### Supported Benchmarks
![Benchmarking Results](./assets/leaderboard.png)

- **DeepResearchBench**: Comprehensive research evaluation
- **ResearchQA**: Question-answering with citation verification  
- **DeepConsult**: Consulting-style analysis tasks

### EDR-200 Dataset

The **[EDR-200 dataset](https://huggingface.co/datasets/Salesforce/EDR-200)** contains 201 complete agentic research trajectories generated by Enterprise Deep Research—99 queries from DeepResearch Bench and 102 queries from DeepConsult. Unlike prior benchmarks that only capture final outputs, these trajectories expose the full reasoning process across search, reflection, and synthesis steps, enabling fine-grained analysis of agentic planning and decision-making dynamics.

### Running Benchmarks
Refer to our detailed [benchmarking guide](benchmarks/README.md).

### Development Setup
```bash
# Testing
python -m pytest tests/
python test_agents.py

# Code quality
black src/ services/ benchmarks/
mypy src/ services/
flake8 src/ services/ benchmarks/

# Development server
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
cd ai-research-assistant && npm run dev
```

## 📁 Project Structure

```text
enterprise-deep-research/
├── ai-research-assistant/       # React frontend
├── benchmarks/                  # Evaluation framework
├── src/                        # Core research engine
│   ├── agent_architecture.py   # Multi-agent orchestration
│   ├── graph.py               # LangGraph workflow definitions
│   ├── state.py               # Research state management
│   ├── simple_steering.py     # Steering & task management
│   ├── steering_integration.py # Steering integration layer
│   ├── prompts.py             # Agent prompts & templates
│   ├── configuration.py       # Agent configuration
│   ├── utils.py               # Utility functions
│   ├── visualization_agent.py # Visualization generation
│   └── tools/                 # Research tools & MCP integration
├── services/                   # Backend services (research, analysis, parsing)
├── routers/                    # FastAPI endpoints
├── models/                     # Data schemas
├── app.py                      # Main FastAPI application
├── llm_clients.py              # LLM provider clients
├── session_store.py            # Session management
└── requirements.txt            # Python dependencies
```

## 📜 License & Citation

Licensed under [Apache 2.0](./LICENSE.txt).

```bibtex
@article{prabhakar2025enterprisedeepresearch,
  title={Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics},
  author={Prabhakar, Akshara and Ram, Roshan and Chen, Zixiang and Savarese, Silvio and Wang, Frank and Xiong, Caiming and Wang, Huan and Yao, Weiran},
  journal={arXiv preprint arXiv:2510.17797},
  year={2025}
}
```

**Acknowledgments**: Built on [LangGraph](https://github.com/langchain-ai/langgraph), [Tavily](https://tavily.com), [React](https://reactjs.org/), [Tailwind CSS](https://tailwindcss.com/), and [FastAPI](https://fastapi.tiangolo.com/).
