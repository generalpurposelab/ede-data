# Ede Python library

[[Blog]]()
[[Paper]]()
[[Data card]]()
[[Yoruba example]]()

The Ede Python library automates the generations of instruction fine-tuning datasets in low-resource languages. PyPI package coming soon. Using GPT-4, it generates at 15.7s per generation using the API simply and 0.55s per generation (~15 hours) using batch processing.

## Setup

The full API for this library is coming soon. For now, you can clone the [repository](https://github.com/generalpurposelab/ede-data), install requirements and run the `run.py` script.

```sh
pip install -r requirements.txt
python run.py
```

 The following parameters are required:

```python
import Ede

api_key = "" # api key
model={"provider": "", "model": ""} # accepts openai and anthropic for provider
target_language = "" # target language e.g. Yoruba
data_dir="data" # data directory (defaults to data). should contain input, output and schemas
size=100 # dataset size (defaults to 100)

pipeline = Ede(
    api_key=api_key, 
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