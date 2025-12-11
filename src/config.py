# Fileserver data location
SERVER_STATIC_DIR = "src/static"
SERVER_APP_STATIC_DIR = "app/static"
SERVER_APP_RAG_INPUT = SERVER_APP_STATIC_DIR + "/rag_input" # path used for fileserver access

# Set to empty string if you dont use Obsidian
# Note: You can sync obsidian with Nextcloud, Dropbox, etc. to enable cloud integration of your notes.
DIRECTORY_OBSIDIAN_VAULT = "/home/noob/Nextcloud/obsidian"
DIRECTORY_CHAT_HISTORIES = "./chat_histories"
DIRECTORY_VLM_OUTPUT = SERVER_STATIC_DIR + "/minerU_output" # VLM data miner output location
DIRECTORY_MD_PREPROCESSING = SERVER_STATIC_DIR + "/md_preprocessing" # preprocessed markdown files location
DIRECTORY_LLM_PREPROCESSING = SERVER_STATIC_DIR + "/llm_preprocessing" # LLM processed markdown files location
DIRECTORY_RAG_INPUT = SERVER_STATIC_DIR + "/rag_input" # prepared markdown files for RAG embeddings

### RAG related config
# Define obsidian subfolder for RAG documents
DIRECTORY_OBSIDIAN_DOCS = "digital-garden"
DIRECTORY_EMBEDDINGS = "./embeddings"
