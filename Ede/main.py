import os
import time
import asyncio
from dotenv import load_dotenv

from .utils.create_csv import CSVCreator
from .utils.generate_qa import QAGenerator

load_dotenv(dotenv_path='.env.local')
api_key = os.getenv('OPENAI_API_KEY')
DEFAULT_MODEL={"provider": "openai", "model": "gpt-4-turbo-preview"}

class Ede:
    def __init__(self, target_language, model=DEFAULT_MODEL, api_key=api_key, data_dir="data", size=100):
        self.api_key = api_key
        self.model = model["model"]  
        self.provider = model["provider"]
        self.target_language = target_language
        self.size = size
        self.data_dir = data_dir

    def run(self):
        input_schema_file = f"{self.data_dir}/schemas/input_schema.csv"
        output_schema_file = f"{self.data_dir}/schemas/output_schema.csv"
        output_file = f"{self.data_dir}/output/output.csv"
        
        if os.path.exists(output_file):
            print(f"Skipping CSV creation as output.csv already exists.")
        else:
            csv_creator = CSVCreator(input_schema_file, output_schema_file, self.data_dir, self.target_language)
            data = csv_creator.generate_data(self.size)
            csv_creator.save_data(data, output_file)
        qa_generator = QAGenerator(output_file, self.api_key, self.target_language, self.model)
        start_time = time.time()
        asyncio.run(qa_generator.process_output_csv())  
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of qa_generator.process_output_csv(): {execution_time:.2f} seconds")