# Ede Python library

The Ede Python library automates the generations of instruction fine-tuning datasets in low-resource languages. PyPI package coming soon. Using GPT-4, it takes 0.5-1.5s per generation (using batch processing) so ~1 day to create 100,000 generations.

## Setup

The full API for this library is coming soon. For now, you can clone the [repository](https://github.com/generalpurposelab/ede-data), install requirements and run the `run.py` script.

```sh
pip install -r requirements.txt
python run.py
```

Don't forget to add your OpenAI API key to the .env file.

```
OPENAI_API_KEY = "your_openai_api_key"
```

The following parameters are editable. It is possible to run with default settings using only target_language and size.

```python
import Ede

model={"provider": "", "model": ""} # accepts openai and anthropic as providers, although anthropic is less performant as it currently lacks a reliable JSON mode.
target_language = "" # target language e.g. Yoruba
data_dir="data" # data directory (defaults to data). Should contain input, output, schemas, and seeds folders (more info on folder structure below)
size=100 # dataset size (defaults to 100)

pipeline = Ede(
    target_language=target_language, 
    model=model, 
    data_dir=data_dir, 
    size=size, 
)

pipeline.run()
```

## Folder structure

    .
    ├── ...
    ├── data                        # Directory containing data for the project
    │   ├── input                   # Input folder contains input files with column names variable_1, variable_2, ..., variable_n
    |   │   ├── input_1.csv      
    |   │   ├── ...              
    |   │   └── input_n.csv      
    │   ├── output                  # Output folder is where generated output is saved as output.csv
    │   ├── prompts                 # Contains template system and user prompts in .txt format
    │   ├── schemas                 # Contains schemas for input and output files (see below)
    |   │   ├── input_schema.csv    
    |   │   └── output_schema.csv   
    │   ├── seeds                   # Contains seed tasks in .jsonl format (see below)
    |   └────── seed_tasks.jsonl      
    └── ...

### Input schema

| file_name  | task_category | task_description | variables | total |
| --- | --- | --- | --- | --- |
| yosm.csv | classification | sentiment analysis of movie reviews | {""variable_1"": ""movie review"", ""variable_2"": ""sentiment""} | 1501 |

### Output schema

| task_category | task_description | percent | total |
| --- | --- | --- | --- | --- |
| classification | sentiment analysis of movie reviews | categorize or label the provided content into predefined classes or groups | 0.03 |

## Versioning

We are keen for your feedback; please open an [issue](https://github.com/generalpurposelab/ede-data/issues) with questions, bugs, or suggestions.

## Requirements

Python 3.7 or higher.