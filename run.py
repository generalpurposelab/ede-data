from Ede.main import Ede

target_language = "Yoruba"
size=100

pipeline = Ede(
    target_language=target_language, 
    size=size, 
)

pipeline.run()