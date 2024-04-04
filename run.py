from Ede.main import Ede

pipeline = Ede(
    target_language="Yoruba", 
    size=1000
)

pipeline.run()