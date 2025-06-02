import json
import os

# === Disease JSON Handling ===
DISEASE_JSON_PATH = 'utils/disease_data.json'

def load_disease_data():
    with open(DISEASE_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_disease_data(data):
    with open(DISEASE_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def update_disease_field(label, field, value):
    data = load_disease_data()
    for item in data.values():
        if item['label'] == label:
            item[field] = value
            save_disease_data(data)
            return True
    return False

# === Artikel JSON Handling ===
ARTIKEL_JSON_PATH = 'utils/artikels.json'

def load_artikel_data():
    with open(ARTIKEL_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_artikel_data(data):
    with open(ARTIKEL_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def update_artikel_field(label, updates: dict):
    data = load_artikel_data()
    for item in data.values():
        if item['label'] == label:
            item.update(updates)
            save_artikel_data(data)
            return True
    return False
