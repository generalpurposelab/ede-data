import os
import time

from .utils.create_csv import CSVCreator
from .utils.generate_qa_anthropic import QAGeneratorAnthropic
from .utils.generate_qa_openai import QAGeneratorOpenAI

class Ede:
    def __init__(self, api_key, target_language, model, data_dir="data", size=100):
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
        if self.provider.lower() == "openai":
            qa_generator = QAGeneratorOpenAI(output_file, self.api_key, self.target_language, self.model)
        else:
            qa_generator = QAGeneratorAnthropic(output_file, self.api_key, self.target_language, self.model)
        
        start_time = time.time()
        qa_generator.process_output_csv()
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Execution time of qa_generator.process_output_csv(): {execution_time:.2f} seconds")