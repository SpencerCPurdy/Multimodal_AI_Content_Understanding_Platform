Live Demo: https://huggingface.co/spaces/SpencerCPurdy/Multimodal_AI_Content_Understanding_Platform

# Multimodal AI Content Understanding Platform

An enterprise-grade AI platform that processes and analyzes multiple content types—images, audio, video, and text—using state-of-the-art machine learning models. This system enables cross-modal search, intelligent content understanding, automated insights extraction, and natural language Q&A across all processed content, making it a comprehensive solution for multimodal data analysis.

## Overview

This platform leverages cutting-edge AI models to understand and analyze diverse content types through a unified interface. Users can upload various media formats, automatically extract meaningful information, search across different modalities using natural language, ask questions about their content, and generate comprehensive insights. The system maintains a vector database for efficient similarity search and retrieval across all content types.

## Key Features

### Content Processing Capabilities
- **Image Analysis**:
  - Automatic caption generation using BLIP
  - Visual feature extraction with CLIP
  - Content moderation and safety checks
  - Dominant color extraction and sharpness analysis
  - Support for JPG, PNG, GIF formats

- **Audio Processing**:
  - Speech-to-text transcription using Whisper
  - Audio feature extraction (spectral analysis, tempo, pitch)
  - Silence detection and dynamic range analysis
  - Support for WAV, MP3 formats

- **Video Analysis**:
  - Frame-by-frame visual analysis
  - Audio track extraction and transcription
  - Scene change detection
  - Combined multimodal insights
  - Support for MP4, AVI formats

- **Text Understanding**:
  - Semantic embedding generation
  - Key phrase extraction
  - Content moderation
  - Language analysis

### Advanced Features
- **Cross-Modal Search**: Search across all content types using natural language queries
- **Vector Database**: ChromaDB integration for efficient similarity search
- **Content Moderation**: Automated safety checks across all modalities
- **Natural Language Q&A**: Ask questions about processed content with optional GPT-4 integration
- **Insights Generation**: AI-powered analysis across multiple content items
- **Batch Processing**: Handle multiple files with persistent storage

## Technologies Used

### AI Models
- **BLIP** (Salesforce/blip-image-captioning-base): Image captioning
- **CLIP** (openai/clip-vit-base-patch32): Vision-language understanding
- **Whisper** (openai/whisper-base): Audio transcription
- **Sentence Transformers** (all-MiniLM-L6-v2): Text embeddings
- **Toxic-BERT** (unitary/toxic-bert): Content moderation

### Core Technologies
- **Gradio**: Interactive web interface
- **PyTorch**: Deep learning framework
- **ChromaDB**: Vector database for similarity search
- **OpenAI API**: Optional GPT-4 integration for enhanced Q&A
- **Transformers**: Hugging Face model library
- **MoviePy**: Video processing
- **Librosa**: Audio analysis
- **OpenCV**: Computer vision operations

## Running the Application

### On Hugging Face Spaces
The application is deployed and ready to use at this Hugging Face Space:
1. Access the space through the provided URL
2. Upload your content files
3. Select the appropriate content type
4. Process and analyze your content
5. Use search, Q&A, and insights features

### Local Installation
To run locally:

```bash
# Clone the repository
git clone [your-repo-url]
cd multimodal-ai-platform

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will launch at `http://localhost:7860`

## Usage Guide

### 1. Content Processing
- Navigate to the "Content Processing" tab
- Upload a file (image, audio, video, or text)
- Select the content type
- Click "Process Content"
- View extracted information, captions, transcripts, and metadata

### 2. Cross-Modal Search
- Go to the "Cross-Modal Search" tab
- Enter a natural language query (e.g., "find images with dogs")
- Optionally filter by content type
- Click "Search" to find relevant content across all modalities

### 3. Question & Answer
- Access the "Question & Answer" tab
- Enter your question about the processed content
- Optionally specify content IDs to focus the search
- Get AI-powered answers based on your content

### 4. Generate Insights
- Open the "Generate Insights" tab
- Enter comma-separated content IDs
- Click "Generate Insights" for AI analysis
- Receive patterns, relationships, and recommendations

### 5. Content Moderation
- Use the "Content Moderation" tab
- Enter text to check for safety
- Receive safety scores and recommendations

## Example Use Cases

### Media Library Management
- Automatically catalog and tag large media collections
- Search across images, videos, and audio using natural descriptions
- Generate metadata and descriptions for accessibility

### Content Analysis & Research
- Analyze video content for research purposes
- Extract and search through podcast transcriptions
- Cross-reference visual and textual information

### Content Moderation & Safety
- Automated screening of user-generated content
- Multi-modal safety checks for platforms
- Compliance verification for content guidelines

### Educational Applications
- Create searchable educational content libraries
- Generate captions and transcripts for accessibility
- Extract key concepts from multimedia lectures

### Business Intelligence
- Analyze presentation videos and extract insights
- Search through meeting recordings and documents
- Generate summaries from multimodal business content

## API Key Configuration

While the core functionality works without an API key, adding an OpenAI API key enables:
- Enhanced natural language Q&A with GPT-4
- More sophisticated insight generation
- Contextual understanding across content

To use these features:
1. Obtain an API key from OpenAI
2. Enter it in the API key field at the top of the interface
3. The key is used only for the current session

## Performance Considerations

- **File Size Limits**: 
  - Images: Resized to 512x512 for processing
  - Audio: Limited to 300 seconds
  - Video: Limited to 600 seconds
- **Processing Time**: Varies by content type and size
- **GPU Acceleration**: Automatically uses CUDA if available
- **Batch Processing**: Process multiple files sequentially

## Data Privacy & Storage

- Content is processed locally or on your Hugging Face Space
- Vector embeddings are stored in ChromaDB for search functionality
- Original files are not permanently stored after processing
- API keys are session-only and never logged

## Troubleshooting

**Common Issues**:
- **"Model loading failed"**: Ensure sufficient memory/GPU resources
- **Slow processing**: Normal for video files; consider using shorter clips
- **Search returns no results**: Ensure content has been processed first
- **Transcription errors**: Check audio quality and language

**Performance Tips**:
- Use GPU-enabled spaces for faster processing
- Process shorter video clips for quicker results
- Batch similar content types together

## Technical Architecture

The platform implements a modular architecture:
- **ModelManager**: Handles loading and caching of AI models
- **ContentProcessors**: Specialized processors for each modality
- **VectorDatabase**: ChromaDB integration for similarity search
- **LLMHandler**: Optional GPT-4 integration
- **GradioInterface**: User interface management

## Future Enhancements

Planned improvements include:
- Support for additional file formats
- Real-time streaming analysis
- Multi-language support
- Enhanced video understanding with temporal models
- API endpoint for programmatic access
- Export functionality for processed data

## License

This project is licensed under the MIT License, allowing for both personal and commercial use with attribution.

## Author

Spencer Purdy
