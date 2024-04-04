# Ede Python library

The Ede Python library automates the generations of instruction fine-tuning datasets in low-resource languages. PyPI package coming soon. Using GPT-4, it takes 0.5-1.5s per generation (using batch processing).

## Setup

The full API for this library is coming soon. For now, you can clone the [repository](https://github.com/generalpurposelab/ede-data), install requirements and run the `run.py` script.

```sh
pip install -r requirements.txt
python run.py
```

 The following parameters are required:

```python
import Ede

model={"provider": "", "model": ""} # accepts openai and anthropic for provider
target_language = "" # target language e.g. Yoruba
data_dir="data" # data directory (defaults to data). should contain input, output and schemas
size=100 # dataset size (defaults to 100)

pipeline = Ede(
    target_language=target_language, 
    model=model, 
    data_dir=data_dir, 
    size=size, 
)

pipeline.run()
```

## Versioning

We are keen for your feedback; please open an [issue](https://github.com/generalpurposelab/ede-data/issues) with questions, bugs, or suggestions.

## Requirements

Python 3.7 or higher.