# self-instruct-global

self-instruct-global automates the generations of multilingual instruction fine-tuning datasets. 

``
from InstructGlobal.main import InstructGlobal

api_key = "" # api key
model={"provider": "", "model": ""} # accepts openai and anthropic for provider
target_language = "" # target language e.g. Yoruba
data_dir="data" # data directory (defaults to data). should contain input, output and schemas
size=100 # dataset size (defaults to 100)

pipeline = InstructGlobal(
    api_key=api_key, 
    target_language=target_language, 
    model=model, 
    data_dir=data_dir, 
    size=size, 
)

pipeline.run()
``