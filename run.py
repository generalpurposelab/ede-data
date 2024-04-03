from Ede.main import Ede

pipeline = Ede(
    target_language="Yoruba", 
    size=100
)

pipeline.run()