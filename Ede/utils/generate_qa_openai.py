# rate limits: https://cookbook.openai.com/examples/how_to_handle_rate_limits
# parallel processing: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
# rate limits: https://platform.openai.com/docs/guides/rate-limits/error-mitigation?context=tier-free

import json
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
import asyncio
import aiohttp

class QAGeneratorOpenAI:
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
        total_rows = len(df)
        completed_rows = 0

        async with aiohttp.ClientSession() as session:
            tasks = []
            for index, row in df.iterrows():
                if pd.isna(row['question']) or pd.isna(row['answer']):
                    task = asyncio.ensure_future(self.process_row(row))
                    tasks.append(task)
                else:
                    completed_rows += 1

            results = await asyncio.gather(*tasks)

            for index, (question, answer) in enumerate(results):
                df.at[df.index[index], 'question'] = question
                df.at[df.index[index], 'answer'] = answer
                completed_rows += 1
                print(f"Completed {completed_rows}/{total_rows} rows")

        # Save the updated DataFrame to the same output file
        df.to_csv(self.output_file, index=False)
        print(f"Updated data saved to {self.output_file}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    async def attempt_create_message(self, session, model, system_content, prompt):
        model_id = model.get("model")
        async with session.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.client.api_key}"},
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
            }
        ) as response:
            response_json = await response.json()
            return response_json

    async def process_prompt(self, user_prompt, system_prompt, model):
        max_attempts = 3
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
                        print(f"Attempt {attempt}: Empty question or answer, retrying...")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on attempt {attempt}: {str(e)}, retrying...")
                attempt += 1
        return "", ""