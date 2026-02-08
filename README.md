# LLM_feature_gen



**LLM Feature Gen** is a Python library for **discovering and generating interpretable features** from unstructured data using Large Language Models (LLMs).  
The library provides high-level utilities for:
- Discovering human-interpretable features from sets of images,
- Integrating prompts and model outputs into structured JSON representations,
- - Generating new feature representations automatically from raw multimodal data,
e.g., creating structured tables for downstream models,


---

## Module: `discover`

The `discover` module focuses on **feature discovery** — identifying interpretable, discriminative visual or textual properties using an LLM.

### ✅ What it does
Given a folder of images and a prompt, the library:
1. Converts each image into Base64 format,  
2. Sends them to an LLM,  
3. Receives a structured JSON response describing the discovered features,  
4. Automatically saves the output to a  JSON file in `outputs/`.

---

## 📂 Project Structure
```text
LLM_feature_gen/
├─ src/
│  └─ LLM_feature_gen/
│     ├─ init.py
│     ├─ discover.py                # High-level orchestration for feature discovery
│     ├─ providers/
          ├─ openai_provider.py     # OpenAI API wrapper
│         ├─ local_provider.py      # Local LLM wrapper
│     ├─ prompts/
│     │   ├─ discovery_prompt.txt   # Default reasoning prompt
          ├─ generation_prompt.txt  # Default feature generation prompt
│     ├─ utils/
│     │   └─ image.py               # Image → base64 conversion
│     └─ tests/
│        └─ test_discover.py
├─ outputs/                         # Automatically generated feature JSONs
├─ pyproject.toml
└─ README.md
```

---

## ⚙️ Installation

Clone or download the repository, then install in editable mode:

```bash
pip install -e .
```

## 🧪 Running Tests

The project uses pytest. You don’t need external services (no network calls are made during tests), and heavy video tooling is stubbed out.

Quick start from the repository root (Windows PowerShell shown, works similarly on macOS/Linux):

```powershell
# 1) (Recommended) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On macOS/Linux: source .venv/bin/activate

# 2) Install the package in editable mode
pip install -e .

# 3) Install test runner
pip install -U pytest

# 4) Run the test suite
python -m pytest -q
```

Useful commands:
- Run a single test file:
  ```powershell
  python -m pytest -q src\tests\test_discovery.py
  ```
- Run tests with verbose output:
  ```powershell
  python -m pytest -vv
  ```

Notes:
- Tests create and use temporary directories; they do not modify your repository files.
- Video-related utilities are monkeypatched/stubbed in tests, so `ffmpeg` is not required to run the suite.
- Environment variables for Azure OpenAI are not required for tests because a fake provider is used.

## 🔑 Environment Setup for OpenAI API

Create a .env file in the project root

##  Example: Discover Features from Images
```python
from LLM_feature_gen.discover import discover_features_from_images
# Folder with your example images
image_folder = "discover_images"

# Run feature discovery
result = discover_features_from_images(
    image_paths_or_folder=image_folder,
    as_set=True,  # analyze all images jointly
)

print(result)
```
This will:
- Read all .jpg/.png images from discover_images/
- the default prompt (prompts/image_discovery_prompt.txt)
- Send them to your LLM provider
- Save the results to outputs/discovered_features_<timestamp>.json

Example saved JSON:
```json
{
  "proposed_features": [
    {
      "feature": "has visible handle",
      "description": "Some objects include handles, others do not.",
      "possible_values": ["present", "absent"]
    },
    {
      "feature": "color tone",
      "description": "Images vary between metallic and earthy color palettes.",
      "possible_values": ["metallic", "matte", "bright", "dark"]
    }
  ]
}
```