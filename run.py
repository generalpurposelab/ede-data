from InstructGlobal.main import InstructGlobal

api_key = ""
model={"provider": "openai", "model": "gpt-4-turbo-preview"}
target_language = "Yoruba"
data_dir="data"
size=10

pipeline = InstructGlobal(
    api_key=api_key, 
    target_language=target_language, 
    model=model, 
    data_dir=data_dir, 
    size=size, 
)

pipeline.run()