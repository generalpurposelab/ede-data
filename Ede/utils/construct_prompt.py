import csv
import json
import random

class PromptConstructor:
    def __init__(self, input_schema_file, output_schema_file, data_dir):
        self.data_dir = data_dir
        with open(input_schema_file, 'r') as file:
            self.input_schema = list(csv.DictReader(file))
        with open(output_schema_file, 'r') as file:
            self.output_schema = list(csv.DictReader(file))
        with open(data_dir + "/seeds/seed_tasks.jsonl", 'r') as file:
            self.seeds = [json.loads(line) for line in file]
        self.source_counts = {item['file_name']: int(item['total']) for item in self.input_schema}
        self.category_sources = {}
        for item in self.input_schema:
            category = item['task_category']
            source_file = item['file_name']
            if category not in self.category_sources:
                self.category_sources[category] = []
            self.category_sources[category].append(source_file)
        self.task_descriptions = {} 
        self.variable_names = {}  
        for item in self.input_schema:
            category = item['task_category']
            source_file = item['file_name']
            self.variable_names[(category, source_file)] = json.loads(item['variables'])
        self.input_data = {}
        for item in self.input_schema:
            file_path = f"data/input/{item['file_name']}"
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.input_data[item['file_name']] = list(reader)
        self.template_dir = data_dir + "/prompts"

    def fetch_input_row_data(self, source_file, task_category):
        if source_file == 'self-instruct':
            yield {}
        else:
            with open(f"{self.data_dir}/input/{source_file}", 'r') as file:
                reader = csv.DictReader(file)
                key = (task_category, source_file)
                if key in self.variable_names:
                    variable_names_mapping = self.variable_names[key]
                    for row in reader:
                        yield {variable_names_mapping.get(f"variable_{i+1}", f"variable_{i+1}"): value for i, value in enumerate(row.values())}
                    while True:
                        yield {}
                else:
                    for row in reader:
                        yield row

    def load_prompts(self, source):
        system_prompt = f"{self.template_dir}/system.txt"
        user_prompt = f"{self.template_dir}/prompt.txt"
        with open(system_prompt, 'r', encoding='utf-8') as file:
            system_message = file.read()
        with open(user_prompt, 'r', encoding='utf-8') as file:
            user_prompt_message = file.read()
        return user_prompt_message, system_message

    def construct_prompts(self, task_category, source, variable_values, target_language, task_description):
        user_prompt, system_prompt = self.load_prompts(source)
        
        key = (task_category, source)  
        if key in self.variable_names:
            variable_names_mapping = self.variable_names[key]
            if len(variable_values) == 1:
                variable_name = next(iter(variable_names_mapping.values()))
                context_str = f"{variable_name}: {next(iter(variable_values.values()))}\n"
            else:
                context_str = "".join([f"{variable_names_mapping.get(f'variable_{i+1}', f'variable_{i+1}')}: {value}\n" for i, value in enumerate(variable_values.values())])
        else:
            context_str = ""
            variable_names_mapping = {}

        category_seeds = [seed for seed in self.seeds if seed['category'] == task_category]
        if len(category_seeds) < 2:
            raise ValueError(f"Not enough seeds for category '{task_category}'. Please add more seeds or handle this case.")
        seed_1, seed_2 = random.sample(category_seeds, 2)
        seed_1_str = json.dumps({"question": seed_1['question'], "answer": seed_1['answer']})
        seed_2_str = json.dumps({"question": seed_2['question'], "answer": seed_2['answer']})
        
        if source == "self-instruct" or context_str == "":
            user_context = ""
            system_context = ""
        else:
            user_context = f"\n\nIncorporate this context:\n<context>\n{context_str}</context>\n\n"
            system_context = "\n5. Incorporate the context provided into your answer." # It has been verified for accuracy and spelling so you can draw from it verbatim as appropriate
        
        user_prompt = user_prompt.format(
            task_category=task_category, 
            task_description=task_description,  
            context=user_context,  
            seed_1=seed_1_str,
            seed_2=seed_2_str
        )
        system_prompt = system_prompt.format(
            language=target_language, 
            context=system_context
        )
        return user_prompt, system_prompt