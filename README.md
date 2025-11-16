# SME Indian Constitution & Rights 🏛️

An intelligent constitutional law assistant with AI-powered content generation, quiz creation, and **video generation** capabilities.

## ✨ Features

### 🤖 AI-Powered Assistance
- Expert knowledge on Indian Constitution and Rights
- Intelligent conversation with context awareness
- RAG-based responses using constitutional documents

### 📝 Content Generation
- **Quiz Generation**: Create custom quizzes with MCQs, fill-in-the-blanks, descriptive questions
- **Document Export**: Generate PDF, DOCX, and PPTX documents
- **Email Integration**: Send generated content via email

### 🎴 **NEW: Interactive Flashcards**
- **Study Cards**: Create Q&A flashcards for constitutional topics
- **Interactive Learning**: Click to flip cards and reveal answers
- **Progress Tracking**: Mark cards as "Got it!" and track study progress
- **Constitutional Focus**: Specialized cards for articles, rights, and case law

### 🎴 **NEW: Interactive Flashcards**
- **Study Cards**: Create Q&A flashcards for constitutional topics
- **Interactive Learning**: Click to flip cards and reveal answers
- **Progress Tracking**: Mark cards as "Got it!" and track study progress
- **Constitutional Focus**: Specialized cards for articles, rights, and case law

### 🎬 **Video Generation**
- **Educational Videos**: Create 2-2.5 minute videos on constitutional topics
- **Automatic Narration**: Text-to-speech using Sarvam API
- **Professional Slides**: Constitutional-themed templates
- **Complete Assembly**: Automated video composition with MoviePy

### 🔍 Advanced Features
- **Web Search Integration**: Real-time information retrieval
- **File Upload Support**: Process PDF, DOCX, PPTX, and text files
- **Multi-interface**: Web UI (Streamlit), CLI client, and REST API
- **Session Management**: Persistent conversations and history

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Required API keys (see Environment Variables section)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SME-Indian-Constitution-And-Rights
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Required
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   
   # For video generation
   SARVAM_API_KEY=your_sarvam_tts_api_key_here
   
   # Optional
   DATABASE_URL=sqlite:///./sme_constitution_2.db
   NVIDIA_GUARDRAILS_API_KEY=your_nvidia_api_key_here
   ```

4. **Initialize the database**
   ```bash
   python -c "from db_models.crud_operations import ensure_db; ensure_db()"
   ```

### Running the Application

**Option 1: Web Interface (Recommended)**
```bash
# Terminal 1: Start the backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start the frontend
streamlit run frontend/streamlit_app.py
```

**Option 2: CLI Interface**
```bash
# Start backend first
uvicorn main:app --reload --port 8000

# In another terminal
python cli_client.py
```

**Option 3: Direct API**
Access the API documentation at: `http://localhost:8000/docs`

## 🎴 Flashcard Usage

### Through Web Interface
1. Open Streamlit app (`http://localhost:8501`)
2. Type requests like:
   - "Create flashcards for Article 21"
   - "Make study cards for fundamental rights"
   - "Generate flashcards about constitutional amendments"
3. Interactive flashcard session will appear below the chat
4. Click cards to flip between questions and answers
5. Use navigation buttons to move between cards
6. Mark cards as "Got it!" to track progress

### Through CLI
```bash
# In the CLI client, type:
"Create flashcards for directive principles"
```

### Flashcard Features
- **Interactive Flipping**: Click to reveal answers
- **Navigation**: First, Previous, Next, Last buttons
- **Progress Tracking**: "Got it!" button to mark completed cards
- **Study Sessions**: Complete sets with progress percentage
- **Constitutional Themes**: Styled with legal document aesthetics

## 🎥 Video Generation Usage

### Through Web Interface
1. Open Streamlit app (`http://localhost:8501`)
2. Type requests like:
   - "Create a video about Article 21"
   - "Generate a video explaining fundamental rights"
   - "Make a video on right to education"

### Through CLI
```bash
# In the CLI client, type:
"Create a video about constitutional amendments"
```

### Through API
```bash
curl -X POST "http://localhost:8000/generate-video" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "topic=Article 21&duration=150&style=educational&include_examples=true&session_id=your_session_id"
```

### Video Features
- **Duration**: 2-2.5 minutes (customizable)
- **Output**: MP4 format, 1920x1080 resolution
- **Audio**: Professional text-to-speech narration
- **Visuals**: Constitutional-themed slides with Indian tricolor accents
- **Content**: Structured educational content with examples

## 📁 Project Structure

```
├── main.py                 # FastAPI application
├── cli_client.py           # Command-line interface
├── chunking_and_embedding.py # RAG and vector operations
├── requirements.txt        # Dependencies
├── README.md              # This file
│
├── frontend/
│   ├── streamlit_app.py   # Streamlit web interface
│   ├── flashcard_component.py # 🎴 Interactive flashcard display
│   └── styles.css         # UI styling
│
├── db_models/
│   ├── models.py          # SQLAlchemy database models
│   └── crud_operations.py # Database operations
│
├── utils/
│   ├── gemini_chain.py    # LLM integration and intent detection
│   ├── langchain_orchestrator.py # Tool orchestration
│   ├── guardrails.py      # Input safety and filtering
│   └── memory_store.py    # Session memory management
│
├── langchain_tools/
│   ├── flashcard_generator/ # 🎴 NEW: Interactive flashcard tools
│   │   ├── flashcard_generation_tool.py # Main tool integration
│   │   └── __init__.py                   # Package initialization
│   ├── video_generator/   # 🎬 Video generation tools
│   │   ├── video_generation_tool.py # Main tool integration
│   │   ├── tts_handler.py           # Sarvam TTS integration
│   │   ├── slide_template_manager.py # Slide creation
│   │   ├── video_composer.py        # MoviePy video assembly
│   │   └── templates/               # Video templates
│   ├── document_exporter/ # Document generation
│   ├── content_generator/ # Content creation tools
│   └── email_agent/       # Email functionality
│
├── preprocess/
│   └── preprocessor.py    # Text cleaning and processing
│
├── data/                  # Input documents and datasets
├── cleaned_data/          # Processed text files
├── generated_videos/      # 🎬 Output directory for videos
└── agent_data/           # Session files and uploads
```

## 🔧 Environment Variables

### Required
- `GEMINI_API_KEY`: Google Gemini API key for LLM functionality
- `OPENAI_API_KEY`: OpenAI API key for embeddings
- `PINECONE_API_KEY`: Pinecone API key for vector database

### Video Generation
- `SARVAM_API_KEY`: Sarvam API key for text-to-speech (required for video narration)

### Optional
- `DATABASE_URL`: Database connection string (default: SQLite)
- `NVIDIA_GUARDRAILS_API_KEY`: NVIDIA API for advanced content filtering

## 🎨 Video Customization

### Templates
Video templates are located in `langchain_tools/video_generator/templates/`:
- Constitutional color scheme (saffron, white, green, navy)
- Professional layouts with Indian constitutional themes
- Customizable slide structures

### Audio Settings
- Voice: Meera (Hindi-English mix) via Sarvam API
- Sample Rate: 22,050 Hz
- Format: WAV/MP3 compatible

### Video Output
- Format: MP4 (H.264)
- Resolution: 1920x1080 (Full HD)
- Frame Rate: 24 FPS
- Typical file size: 50-100 MB for 2.5-minute video

## 🧪 Example Requests

### Constitutional Content
- "Explain Article 21 in detail"
- "What are the fundamental rights?"
- "Difference between fundamental rights and directive principles"

### Quiz Generation
- "Create a quiz on fundamental rights with 10 MCQs"
- "Generate 5 descriptive questions about constitutional amendments"

### 🎴 Flashcard Generation
- "Create flashcards for fundamental rights"
- "Make study cards about Article 14-18"
- "Generate flashcards on constitutional remedies"

### 🎬 Video Generation
- "Create a video about the right to privacy"
- "Make an educational video explaining Article 356"
- "Generate a video on constitutional remedies with examples"

### Document Export
- "Export this content as a PDF"
- "Create a PowerPoint presentation about directive principles"

## 🛠️ Development

### Adding New Video Templates
1. Create template in `langchain_tools/video_generator/templates/`
2. Update `SlideTemplateManager` to handle new template
3. Modify `video_generation_tool.py` to use new template

### Extending Video Features
- Add new narration voices in `tts_handler.py`
- Customize video transitions in `video_composer.py`
- Create new slide layouts in `slide_template_manager.py`

### Testing Video Generation
```bash
# Test TTS functionality
python -c "from langchain_tools.video_generator.tts_handler import SarvamTTSHandler; handler = SarvamTTSHandler(); print('TTS Ready:', handler.api_key is not None)"

# Test slide generation
python -c "from langchain_tools.video_generator.slide_template_manager import SlideTemplateManager; manager = SlideTemplateManager(); print('Template created:', manager.create_constitutional_template())"
```

## 📊 Performance Notes

### Video Generation Times
- Script generation: 5-10 seconds
- TTS audio creation: 10-20 seconds
- Slide rendering: 5-10 seconds
- Video assembly: 20-30 seconds
- **Total**: ~45-70 seconds for a 2.5-minute video

### Resource Usage
- Memory: ~2-4 GB during video generation
- Storage: ~50-100 MB per generated video
- CPU: Intensive during MoviePy processing

## 🚨 Troubleshooting

### Video Generation Issues
1. **No audio in video**: Check `SARVAM_API_KEY` configuration
2. **Video composition fails**: Ensure MoviePy and FFmpeg are properly installed
3. **Slides not rendering**: Verify PIL/Pillow installation
4. **Long processing times**: Normal for high-quality video generation

### Common Issues
1. **Import errors**: Run `pip install -r requirements.txt`
2. **Database errors**: Run the database initialization command
3. **API errors**: Check environment variables and API key validity

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for constitutional education and legal literacy in India** 🇮🇳