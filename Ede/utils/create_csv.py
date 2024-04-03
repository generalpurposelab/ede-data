import csv
from .construct_prompt import PromptConstructor
from tqdm import tqdm  

class CSVCreator:
    def __init__(self, input_schema_file, output_schema_file, data_dir, target_language):
        self.prompt_constructor = PromptConstructor(input_schema_file, output_schema_file, data_dir)
        self.target_language = target_language  
        
    def generate_data(self, size):
        data = []
        # This dictionary now includes task_description for each category
        category_details = {item['task_category']: {'count': int(float(item['percent']) * size), 'description': item['task_description']} for item in self.prompt_constructor.output_schema}
        
        total_count = sum(detail['count'] for detail in category_details.values())
        remaining_count = size - total_count
        
        categories = list(category_details.keys())
        while remaining_count > 0:
            for category in categories:
                if remaining_count > 0:
                    category_details[category]['count'] += 1
                    remaining_count -= 1
                else:
                    break
        
        for category, detail in tqdm(category_details.items(), desc="Building output.csv"):
            count = detail['count']
            task_description = detail['description']  # Extract task_description for the current category
            for _ in range(count):
                source_file = self.select_source_file(category)
                input_row_data_generator = self.prompt_constructor.fetch_input_row_data(source_file, category)
                
                try:
                    input_row_data = next(input_row_data_generator)
                except StopIteration:
                    input_row_data = {}
                
                # Pass task_description to construct_prompts
                user_prompt, system_prompt = self.prompt_constructor.construct_prompts(category, source_file, input_row_data, self.target_language, task_description)
                
                row = {
                    'question': '',
                    'answer': '',
                    'user_prompt': user_prompt,
                    'system_prompt': system_prompt,
                    'task_category': category,
                    'source': source_file
                }
                data.append(row)
        
        return data

    def select_source_file(self, category):
        if category in self.prompt_constructor.category_sources and self.prompt_constructor.category_sources[category]:
            source_file = self.prompt_constructor.category_sources[category][0]
            if self.prompt_constructor.source_counts[source_file] > 0:
                self.prompt_constructor.source_counts[source_file] -= 1
            else:
                self.prompt_constructor.category_sources[category].pop(0)
                if self.prompt_constructor.category_sources[category]:
                    source_file = self.prompt_constructor.category_sources[category][0]
                    self.prompt_constructor.source_counts[source_file] -= 1
                else:
                    source_file = 'self-instruct'
        else:
            source_file = 'self-instruct'
        
        return source_file

    def save_data(self, data, output_file):
        fieldnames = ['question', 'answer', 'user_prompt', 'system_prompt', 'task_category', 'source']
        sorted_data = sorted(data, key=lambda x: x['task_category'])
        
        # Count 'self-instruct' and other context values
        self_instruct_count = sum(1 for row in data if row['source'] == 'self-instruct')
        context_count = len(data) - self_instruct_count
        
        with open(output_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_data)
        print(f"Output csv initialised with {self_instruct_count} self-instruct values and {context_count} context values")