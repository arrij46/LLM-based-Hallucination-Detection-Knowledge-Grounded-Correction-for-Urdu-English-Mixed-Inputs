# 1️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate     

# 2️⃣ Upgrade pip to latest version
python -m pip install --upgrade pip

# 3️⃣ Install required Python packages
pip install transformers
pip install torch sentencepiece pandas

# Or alternatively:
python -m pip install pandas transformers sentencepiece torch

# 4️⃣ Verify installation
python -c "import pandas, transformers, torch, sentencepiece; print('All OK')"

# 5️⃣ List installed packages (optional)
pip list

# 6️⃣ Run Phase 1 preprocessing script
python -m pipeline.stage1_preprocessing


