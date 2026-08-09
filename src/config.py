import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "guardian_environment_news.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "labeled_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORIES = {
    "Climate Policy": [
        "climate policy",
        "environmental regulation", "emissions trading", "carbon tax", 
        "climate summit", "legislation", "parliament", "cop26", "cop27", "cop28",
        "net zero target", "decarbonisation policy"
    ],
    "Biodiversity": [
        "species", "wildlife", "forest", "biodiversity", "ocean", 
        "animal", "extinction", "conservation", "habitat", "ecosystem",
        "marine life", "endangered species", "deforestation", "rewilding"
    ],
    "Renewable Energy": [
        "solar", "wind", "renewable", "electric vehicle", "battery", 
        "clean energy", "grid", "hydrogen", "photovoltaic", "heat pump",
        "wind farm", "solar panel", "energy transition", "offshore wind", "geothermal"
    ],
    "Disasters & Extreme Weather": [
        "flood", "wildfire", "drought", "hurricane", "heatwave", 
        "storm", "disaster", "typhoon", "extreme weather", "torrential rain",
        "sea level rise", "flash flood", "state of emergency"
    ],
}