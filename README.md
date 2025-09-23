# Enterprise Deep Research

A comprehensive AI-powered research assistant with both Python backend and React frontend components.

## 🛠️ Installation

### Prerequisites

- Python 3.11
- Node.js v20.9.0
- npm package manager

### First Time Setup

Follow these steps to set up the project from scratch:

#### 1. Create Python Virtual Environment

```bash
python -m venv venv
```

#### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

#### 3. Install Python Dependencies

```bash
python -m pip install -r requirements.txt
```

#### 4. Environment Configuration

Set up `.env` files in both locations:

- **Root folder**: Create `.env` file in the project root directory
- **AI Research Assistant folder**: Create `.env` file in the `/ai-research-assistant` folder

> **Note**: Make sure to configure the necessary environment variables in both `.env` files according to your setup requirements.

#### 5. Frontend Setup

Navigate to the AI research assistant directory and install dependencies:

```bash
cd ai-research-assistant && npm install && npm run build
```

#### 6. Start the Servers

You need to run both servers simultaneously. Open two terminal windows/tabs:

##### Terminal A - Frontend Server

```bash
cd ai-research-assistant
npm start
```

##### Terminal B - Python Backend Server

```bash
# Make sure you're in the root directory and virtual environment is activated
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🚀 Quick Start (After Initial Setup)

For subsequent runs after the initial setup:

1. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Start both servers as described in step 6 above.

### Project Structure

- `ai-research-assistant/` - React frontend application
- `app.py` - Main Python backend application
- `requirements.txt` - Python dependencies
- `mcp_servers/` - MCP (Model Context Protocol) server implementations
- `src/` - Additional source code and tools

### Troubleshooting

- Ensure all environment variables are properly configured in both `.env` files
- Check that all dependencies are installed correctly
- Make sure both servers are running on their respective ports
- Verify that the virtual environment is activated when running Python commands

## 📊 Benchmarking with Enterprise Deep Research Agent

For detailed benchmarking guidelines, refer to `benchmarks/README.md`