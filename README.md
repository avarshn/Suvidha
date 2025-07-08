# Suvidha - Intelligent Shopping Assistant



### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root with:
```bash
# Search API Configuration
SEARCH_API_KEY=your_search_api_key_here
SERP_API_KEY=your_serp_api_key_here

# Reddit API Configuration (Required for comment fetching)
# Get these from https://www.reddit.com/prefs/apps/
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=python:suvidha.shopping.assistant:v1.0 (by /u/your_username)

# Groq API Configuration (For speech-to-text)
GROQ_API_KEY=your_groq_api_key_here
```

#### Setting up Reddit API:
1. Go to https://www.reddit.com/prefs/apps/
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in the name and description
5. Set redirect URI to `http://localhost:8080` (not used but required)
6. Copy the client ID (under the app name) and client secret

### 4. Run the Application
```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

## Features
- **Intelligent Chat**: AI-powered shopping assistant
- **Reddit Integration**: Fetches real user reviews and discussions
- **Product Discovery**: Automatic product search and comparison
- **Speech Input**: Voice-to-text functionality
- **Preference Tracking**: Visual preference graph
- **Multi-tab Interface**: Organized chat, products, and posts views