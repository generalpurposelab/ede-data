import os
from dotenv import load_dotenv

from Ede.main import Ede

load_dotenv(dotenv_path='.env.local')

api_key = os.getenv('OPENAI_API_KEY')
model={"provider": "openai", "model": "gpt-4-turbo-preview"}
target_language = "Yoruba"
data_dir="data"
size=100

pipeline = Ede(
    api_key=api_key, 
    target_language=target_language, 
    model=model, 
    data_dir=data_dir, 
    size=size, 
)

pipeline.run()