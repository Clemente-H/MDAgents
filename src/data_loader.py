import json
import csv

def load_data(dataset):
    if dataset = 'mycsvdataset':
        return load_csv_data('./data/mycsvdataset/preguntas.csv')
    else:
        test_qa = []
        examplers = []
        test_path = f'./data/{dataset}/test.jsonl'
        train_path = f'./data/{dataset}/train.jsonl'

        with open(test_path, 'r') as file:
            for line in file:
                test_qa.append(json.loads(line))

        with open(train_path, 'r') as file:
            for line in file:
                examplers.append(json.loads(line))

        return test_qa, examplers

def load_csv_data(csv_path):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t') # Ajustar el delimitador según tu CSV.
        for row in reader:
            # row keys: 'pregunta', 'respuesta_correcta', 'nombre_imagen', 'categoria_1', 'categoria_2', 'ruta'
            question_text = row['pregunta']
            correct_answer = row['respuesta_correcta'].strip().upper() # Por ej. 'A', 'B', etc.

            # Extraemos las opciones de la pregunta
            # Suponemos un formato: ... (a) OpcionA (b) OpcionB (c) OpcionC (d) OpcionD
            options = parse_options_from_question(question_text) 
            
            # Obtenemos el texto de la opción correcta
            answer_text = options.get(correct_answer, "")

            sample = {
                'question': question_text,
                'answer_idx': correct_answer,
                'answer': answer_text,
                'options': options,
                'img_path': row['ruta']
            }
            data.append(sample)

    # Suponiendo que no tenemos separacion en train/test en tu CSV, puedes usar data como test_qa
    # y examplers vacio o alguno base. Para test: 
    return data, []

def parse_options_from_question(question):
    # Ejemplo de parsing simple.
    # Buscamos patrones (a) ... (b) ... (c) ... (d) ...
    # Esto se puede mejorar con regex, aquí un ejemplo simple.
    import re
    pattern = r"\((a)\)\s*(.*?)\s*\((b)\)\s*(.*?)\s*\((c)\)\s*(.*?)\s*\((d)\)\s*(.*)"
    match = re.search(pattern, question, re.IGNORECASE)
    if not match:
        # Si no se encuentran las opciones, retornar vacío o algún fallback.
        return {}
    # match groups:
    # group(1) = 'a', group(2) = opcion A, group(4) = opcion B, etc.
    # groups: (a) 2, (b) 4, (c) 6, (d) 8
    options = {
        'A': match.group(2).strip(),
        'B': match.group(4).strip(),
        'C': match.group(6).strip(),
        'D': match.group(8).strip()
    }
    return options