"""
test_package.py
---------------
Tests discover + generate for all modalities using both:
  🔵 OpenAIProvider  (OpenAI / Azure)
  🟢 LocalProvider   (Ollama / vLLM / LM Studio)

Run:
    python test_package.py

Prerequisites — .env file:
    # OpenAI / Azure
    OPENAI_API_KEY=...
    OPENAI_MODEL=gpt-4o
    OPENAI_AUDIO_MODEL=whisper-1

    # Local (Ollama)
    LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1
    LOCAL_OPENAI_API_KEY=ollama
    LOCAL_MODEL_TEXT=llama3
    LOCAL_MODEL_VISION=llava
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from LLM_feature_gen.providers.openai_provider import OpenAIProvider
from LLM_feature_gen.providers.local_provider import LocalProvider

from LLM_feature_gen.discover import (
    discover_features_from_images,
    discover_features_from_texts,
    discover_features_from_tabular,
    discover_features_from_videos,
)
from LLM_feature_gen.generate import (
    generate_features_from_images,
    generate_features_from_texts,
    generate_features_from_tabular,
    generate_features_from_videos,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_OPENAI = Path("outputs/test_openai")
OUT_LOCAL  = Path("outputs/test_local")
OUT_OPENAI.mkdir(parents=True, exist_ok=True)
OUT_LOCAL.mkdir(parents=True, exist_ok=True)

# Discover sample data
IMAGES_DIR   = Path("discover_images")
TEXTS_DIR    = Path("discover_texts")
TABULAR_FILE = Path("discover_tabular/test.csv")
VIDEOS_DIR   = Path("discover_videos")

# Generate data
GEN_IMAGES_DIR  = Path("images")
GEN_TEXTS_DIR   = Path("texts")
GEN_TABULAR_DIR = Path("tabular")
GEN_VIDEOS_DIR  = Path("videos")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pp(label: str, data: object) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1 — Provider initialisation
# ---------------------------------------------------------------------------
_section("1 — Provider initialisation")

try:
    openai_provider = OpenAIProvider()
    print(f"🔵 OpenAIProvider ready  |  model={openai_provider.default_model}  |  azure={openai_provider.is_azure}")
except EnvironmentError as e:
    openai_provider = None
    print(f"⚠️  OpenAIProvider not available: {e}")

try:
    local_provider = LocalProvider()
    print(f"🟢 LocalProvider  ready  |  text={local_provider.text_model}  |  vision={local_provider.vision_model}  |  url={local_provider.base_url}")
except Exception as e:
    local_provider = None
    print(f"⚠️  LocalProvider not available: {e}")


# ---------------------------------------------------------------------------
# 2 — Discover: Texts
# ---------------------------------------------------------------------------
_section("2 — Discover: Texts")

if openai_provider:
    print("🔵 Discovering text features (OpenAI) …")
    result = discover_features_from_texts(
        texts_or_file=TEXTS_DIR,
        provider=openai_provider,
        output_dir=OUT_OPENAI,
        output_filename="discovered_text_features.json",
    )
    _pp("OpenAI result", result)
else:
    print("⏭  Skipped (no OpenAI provider)")

if local_provider:
    print("🟢 Discovering text features (Local) …")
    result = discover_features_from_texts(
        texts_or_file=TEXTS_DIR,
        provider=local_provider,
        output_dir=OUT_LOCAL,
        output_filename="discovered_text_features.json",
    )
    _pp("Local result", result)
else:
    print("⏭  Skipped (no Local provider)")


# ---------------------------------------------------------------------------
# 3 — Discover: Tabular
# ---------------------------------------------------------------------------
_section("3 — Discover: Tabular")

if openai_provider:
    print("🔵 Discovering tabular features (OpenAI) …")
    result = discover_features_from_tabular(
        file_or_folder=TABULAR_FILE,
        text_column="text",
        provider=openai_provider,
        max_rows=10,
        output_dir=OUT_OPENAI,
        output_filename="discovered_tabular_features.json",
    )
    _pp("OpenAI result", result)
else:
    print("⏭  Skipped (no OpenAI provider)")

if local_provider:
    print("🟢 Discovering tabular features (Local) …")
    result = discover_features_from_tabular(
        file_or_folder=TABULAR_FILE,
        text_column="text",
        provider=local_provider,
        max_rows=10,
        output_dir=OUT_LOCAL,
        output_filename="discovered_tabular_features.json",
    )
    _pp("Local result", result)
else:
    print("⏭  Skipped (no Local provider)")


# ---------------------------------------------------------------------------
# 4 — Discover: Images
# ---------------------------------------------------------------------------
_section("4 — Discover: Images")

if openai_provider:
    print("🔵 Discovering image features (OpenAI) …")
    result = discover_features_from_images(
        image_paths_or_folder=IMAGES_DIR,
        provider=openai_provider,
        output_dir=OUT_OPENAI,
        output_filename="discovered_image_features.json",
    )
    _pp("OpenAI result", result)
else:
    print("⏭  Skipped (no OpenAI provider)")

if local_provider:
    print("🟢 Discovering image features (Local) …")
    result = discover_features_from_images(
        image_paths_or_folder=IMAGES_DIR,
        provider=local_provider,
        output_dir=OUT_LOCAL,
        output_filename="discovered_image_features.json",
    )
    _pp("Local result", result)
else:
    print("⏭  Skipped (no Local provider)")


# ---------------------------------------------------------------------------
# 5 — Discover: Videos
# ---------------------------------------------------------------------------
_section("5 — Discover: Videos")

if openai_provider:
    print("🔵 Discovering video features (OpenAI) …")
    result = discover_features_from_videos(
        videos_or_folder=VIDEOS_DIR,
        provider=openai_provider,
        use_audio=True,
        output_dir=OUT_OPENAI,
        output_filename="discovered_video_features.json",
    )
    _pp("OpenAI result", result)
else:
    print("⏭  Skipped (no OpenAI provider)")

if local_provider:
    print("🟢 Discovering video features (Local) …  (use_audio=True requires faster-whisper)")
    result = discover_features_from_videos(
        videos_or_folder=VIDEOS_DIR,
        provider=local_provider,
        use_audio=True,
        output_dir=OUT_LOCAL,
        output_filename="discovered_video_features.json",
    )
    _pp("Local result", result)
else:
    print("⏭  Skipped (no Local provider)")


# ---------------------------------------------------------------------------
# 6 — Generate: Texts
# ---------------------------------------------------------------------------
_section("6 — Generate: Texts")

if openai_provider and (OUT_OPENAI / "discovered_text_features.json").exists():
    print("🔵 Generating text feature values (OpenAI) …")
    csv_paths = generate_features_from_texts(
        root_folder=GEN_TEXTS_DIR,
        discovered_features_path=OUT_OPENAI / "discovered_text_features.json",
        provider=openai_provider,
        output_dir=OUT_OPENAI,
    )
    for cls, path in csv_paths.items():
        print(f"  {cls}: {path}")
else:
    print("⏭  Skipped (run section 2 first or no OpenAI provider)")

if local_provider and (OUT_LOCAL / "discovered_text_features.json").exists():
    print("🟢 Generating text feature values (Local) …")
    csv_paths = generate_features_from_texts(
        root_folder=GEN_TEXTS_DIR,
        discovered_features_path=OUT_LOCAL / "discovered_text_features.json",
        provider=local_provider,
        output_dir=OUT_LOCAL,
    )
    for cls, path in csv_paths.items():
        print(f"  {cls}: {path}")
else:
    print("⏭  Skipped (run section 2 first or no Local provider)")


# ---------------------------------------------------------------------------
# 7 — Generate: Tabular
# ---------------------------------------------------------------------------
_section("7 — Generate: Tabular")

if openai_provider and (OUT_OPENAI / "discovered_tabular_features.json").exists():
    print("🔵 Generating tabular feature values (OpenAI) …")
    csv_paths = generate_features_from_tabular(
        root_folder=GEN_TABULAR_DIR,
        discovered_features_path=OUT_OPENAI / "discovered_tabular_features.json",
        provider=openai_provider,
        output_dir=OUT_OPENAI,
        text_column="text",
        label_column="label",
    )
    for cls, path in csv_paths.items():
        print(f"  {cls}: {path}")
else:
    print("⏭  Skipped (run section 3 first or no OpenAI provider)")

if local_provider and (OUT_LOCAL / "discovered_tabular_features.json").exists():
    print("🟢 Generating tabular feature values (Local) …")
    csv_paths = generate_features_from_tabular(
        root_folder=GEN_TABULAR_DIR,
        discovered_features_path=OUT_LOCAL / "discovered_tabular_features.json",
        provider=local_provider,
        output_dir=OUT_LOCAL,
        text_column="text",
        label_column="label",
    )
    for cls, path in csv_paths.items():
        print(f"  {cls}: {path}")
else:
    print("⏭  Skipped (run section 3 first or no Local provider)")


# ---------------------------------------------------------------------------
# 8 — Generate: Images
# ---------------------------------------------------------------------------
_section("8 — Generate: Images")

if openai_provider and (OUT_OPENAI / "discovered_image_features.json").exists():
    print("🔵 Generating image feature values (OpenAI) …")
    csv_paths = generate_features_from_images(
        root_folder=GEN_IMAGES_DIR,
        discovered_features_path=OUT_OPENAI / "discovered_image_features.json",
        provider=openai_provider,
        output_dir=OUT_OPENAI,
    )
    for cls, path in csv_paths.items():
        print(f"  {cls}: {path}")
else:
    print("⏭  Skipped (run section 4 first or no OpenAI provider)")

if local_provider and (OUT_LOCAL / "discovered_image_features.json").exists():
    print("🟢 Generating image feature values (Local) …")
    csv_paths = generate_features_from_images(
        root_folder=GEN_IMAGES_DIR,
        discovered_features_path=OUT_LOCAL / "discovered_image_features.json",
        provider=local_provider,
        output_dir=OUT_LOCAL,
    )
    for cls, path in csv_paths.items():
        print(f"  {cls}: {path}")
else:
    print("⏭  Skipped (run section 4 first or no Local provider)")


# ---------------------------------------------------------------------------
# 9 — Generate: Videos
# ---------------------------------------------------------------------------
_section("9 — Generate: Videos")

if openai_provider and (OUT_OPENAI / "discovered_video_features.json").exists():
    print("🔵 Generating video feature values (OpenAI) …")
    csv_paths = generate_features_from_videos(
        root_folder=GEN_VIDEOS_DIR,
        discovered_features_path=OUT_OPENAI / "discovered_video_features.json",
        provider=openai_provider,
        output_dir=OUT_OPENAI,
        use_audio=True,
    )
    for cls, path in csv_paths.items():
        print(f"  {cls}: {path}")
else:
    print("⏭  Skipped (run section 5 first or no OpenAI provider)")

if local_provider and (OUT_LOCAL / "discovered_video_features.json").exists():
    print("🟢 Generating video feature values (Local) …")
    csv_paths = generate_features_from_videos(
        root_folder=GEN_VIDEOS_DIR,
        discovered_features_path=OUT_LOCAL / "discovered_video_features.json",
        provider=local_provider,
        output_dir=OUT_LOCAL,
        use_audio=True,
    )
    for cls, path in csv_paths.items():
        print(f"  {cls}: {path}")
else:
    print("⏭  Skipped (run section 5 first or no Local provider)")


# ---------------------------------------------------------------------------
# 10 — Summary
# ---------------------------------------------------------------------------
_section("10 — Summary")

print("📂 OpenAI outputs:")
for f in sorted(OUT_OPENAI.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUT_OPENAI)}  ({f.stat().st_size / 1024:.1f} KB)")

print("\n📂 Local outputs:")
for f in sorted(OUT_LOCAL.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUT_LOCAL)}  ({f.stat().st_size / 1024:.1f} KB)")

