import os
import json
import random
import argparse
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from termcolor import cprint
from src.model_setup import setup_model
from src.difficulty_selector import determine_difficulty
from src.query_processing import (
    process_basic_query, 
    process_intermediate_query, 
    process_advanced_query
)
from src.utils import check_image_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    # parser.add_argument('--difficulty', type=str, default='adaptive')
    # parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--start_index', type=int, default=0)

    args = parser.parse_args()
    df = pd.read_excel(args.dataset)
    #model, client = setup_model(args.model)
    #test_qa, examplers = load_data(args.dataset)

    agent_emoji = [
        '\U0001F468\u200D\u2695\uFE0F', 
        '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', 
        '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', 
        # ... (resto de emojis)
    ]
    random.shuffle(agent_emoji)

    results = []
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S") 

    # Iterar desde el índice de inicio
    for no, sample in tqdm(df.iloc[args.start_index:].iterrows(),total=len(df),initial=args.start_index):
        image_path = os.path.join('../', sample['ruta'])
        image_path = os.path.join('../', sample['ruta'])
        # Check image size
        is_valid_size, file_size = check_image_size(image_path, max_size_mb=4.0)
        if os.path.exists(image_path) and is_valid_size:
            try:
                #difficulty = determine_difficulty(sample['pregunta'], 'adaptive', image_path)
                #print(difficulty)
                final_decision, history = process_advanced_query(sample['pregunta'], model='gpt-4o-mini', args=None, img_path=image_path)
                respuesta = {
                    'no': no,
                    'categoria_1':sample['categoria_1'],
                    'categoria_2':sample['categoria_2'],
                    'pregunta': sample['pregunta'],
                    'true_answer': sample['respuesta_correcta'],
                    'img_path':image_path,
                    'dificulty': 'advanced',
                    'final_decision':final_decision,
                    'predicted_answer':'',
                    'history':history,
                }
            except:
                respuesta = {
                    'no': no,
                    'categoria_1':sample['categoria_1'],
                    'categoria_2':sample['categoria_2'],
                    'pregunta': sample['pregunta'],
                    'true_answer': sample['respuesta_correcta'],
                    'img_path':image_path,
                    'dificulty': 'advanced',
                    'final_decision':'Hubo un big problem, oh oh.',
                    'predicted_answer':'',
                    'history':'no hay, hubo problem xdxd',
                }
            results.append(respuesta)
         # Guardar resultados cada 10 iteraciones
            if (no + 1) % 10 == 0:
                output_file = f'results/{args.model}_{formatted_time}_batch_{no+1}.json'
                with open(output_file, 'w') as file:
                    json.dump(results, file, indent=4)
                print(f"[INFO] Resultados parciales guardados en {output_file}")

            

    # Guardar resultados
    path = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(path):
        os.makedirs(path)


 # Ejemplo: 20241216_153045
    output_file = f'results/{args.model}_{formatted_time}_advancedquery.json'
#    output_file = f'output/{args.model}_{args.dataset}_advancedquery.json'
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"[INFO] Results saved at {output_file}")

if __name__ == "__main__":
    main()