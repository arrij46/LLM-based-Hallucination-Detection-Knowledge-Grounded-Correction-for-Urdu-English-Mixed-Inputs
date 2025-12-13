## Setup Instructions

1. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

2. Upgrade pip
```bash
python -m pip install --upgrade pip
```

3. Install required packages
```bash
pip install pandas transformers sentencepiece torch

# Or alternatively:
pip install transformers
pip install torch sentencepiece pandas
```

4. Verify installation
```bash
python -c "import pandas, transformers, torch, sentencepiece; print('All packages installed successfully!')"
```

5. Run preprocessing
```bash
python -m pipeline.stage1_preprocessing
```
