import json
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
import asyncio
import aiohttp

class QAGenerator:
    def __init__(self, output_file, api_key, target_language, model="gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=api_key)
        self.output_file = output_file
        self.model = model
        self.target_language = target_language

    async def process_row(self, row):
        question, answer = await self.process_prompt(row['user_prompt'], row['system_prompt'], self.model)
        return question, answer

    async def process_output_csv(self):
        df = pd.read_csv(self.output_file)
        # Ensure 'question' and 'answer' columns are of type object (string)
        # df['question'] = df['question'].astype('object')
        # df['answer'] = df['answer'].astype('object')
        total_rows = len(df)
        completed_rows = 0
        batch_size = 100 

        async with aiohttp.ClientSession() as session:
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                tasks = []
                for index, row in df.iloc[start:end].iterrows():
                    if pd.isna(row['question']) or pd.isna(row['answer']):
                        task = asyncio.ensure_future(self.process_row(row))
                        tasks.append(task)
                    else:
                        completed_rows += 1

                results = await asyncio.gather(*tasks)

                for task_index, (question, answer) in enumerate(results):
                    actual_index = start + task_index
                    df.at[df.index[actual_index], 'question'] = question
                    df.at[df.index[actual_index], 'answer'] = answer
                    completed_rows += 1
                    
                print(f"Completed {completed_rows}/{total_rows} rows")
                # Save the updated DataFrame to the same output file after each batch
                df.to_csv(self.output_file, index=False)
                # print(f"Updated data saved to {self.output_file} for rows {start} to {end}")

        print(f"Final data saved to {self.output_file}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    async def attempt_create_message(self, session, model, system_content, prompt):
        model_id = model.get("model")
        async with session.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.client.api_key}"},
            json={
                "model": model_id,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
            }
        ) as response:
            response_json = await response.json()
            return response_json

    async def process_prompt(self, user_prompt, system_prompt, model):
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            try:
                async with aiohttp.ClientSession() as session:
                    model_dict = {"model": model} if isinstance(model, str) else model
                    full_message = await self.attempt_create_message(session, model_dict, system_prompt, user_prompt)
                    output = json.loads(full_message['choices'][0]['message']['content'])
                    question = output.get("question", "")
                    answer = output.get("answer", "")
                    if question and answer:
                        return question, answer
                    else:
                        attempt += 1
                        # print(f"Attempt {attempt}: Empty question or answer, retrying...")
            except json.JSONDecodeError as e:
                # print(f"JSON parsing error on attempt {attempt}: {str(e)}, retrying...")
                attempt += 1
        return "", ""