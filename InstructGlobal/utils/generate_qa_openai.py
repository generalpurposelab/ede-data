import os
import pandas as pd
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI


class QAGeneratorOpenAI:
    def __init__(self, output_file, api_key, target_language, model="gpt-4-turbo-preview", new_output_file=None):
        self.client = OpenAI(api_key=api_key)
        self.output_file = output_file
        self.model = model
        self.target_language = target_language
        self.new_output_file = new_output_file if new_output_file else f"{output_file.split('.csv')[0]}_instruct.csv"

    def process_output_csv(self):
        df = pd.read_csv(self.output_file)
        total_rows = len(df)
        completed_rows = 0
        file_exists = os.path.exists(self.new_output_file)

        for i in range(0, total_rows, 5):
            updated_rows = [] 
            batch = df.iloc[i:i+5]
            for _, row in batch.iterrows():
                question, answer = self.process_prompt(row['user_prompt'], row['system_prompt'], self.model)
                row['question'] = question
                row['answer'] = answer
                updated_rows.append(row)
                completed_rows += 1
                print(f"Completed {completed_rows}/{total_rows} rows")

            updated_df = pd.DataFrame(updated_rows)
            if not file_exists:
                updated_df.to_csv(self.new_output_file, index=False)
                file_exists = True  
            else:
                updated_df.to_csv(self.new_output_file, mode='a', header=False, index=False)

        print(f"Updated data saved to {self.new_output_file}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    def attempt_create_message(self, model, system_content, prompt):
        model_id = model.get("model")  
        return self.client.chat.completions.create(
        model=model_id,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        )
    
    def process_prompt(self, user_prompt, system_prompt, model): 
        max_attempts = 3  
        attempt = 0  
        while attempt < max_attempts:
            try:
                model_dict = {"model": model} if isinstance(model, str) else model
                full_message = self.attempt_create_message(model_dict, system_prompt, user_prompt)
                output = json.loads(full_message.choices[0].message.content)
                question = output.get("question", "")
                answer = output.get("answer", "")
                if question and answer:
                    return question, answer
                else:
                    attempt += 1
                    print(f"Attempt {attempt}: Empty question or answer, retrying...")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on attempt {attempt}: {str(e)}, retrying...")
                attempt += 1
        return "", ""