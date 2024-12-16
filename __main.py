import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from src.model_setup import setup_model
from src.data_loader import load_data
from src.query_processing import create_question
from src.difficulty_selector import determine_difficulty
from src.query_processing import (
    process_basic_query, 
    process_intermediate_query, 
    process_advanced_query
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='medqa')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--difficulty', type=str, default='adaptive')
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    model, client = setup_model(args.model)
    test_qa, examplers = load_data(args.dataset)

    agent_emoji = [
        '\U0001F468\u200D\u2695\uFE0F', 
        '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', 
        '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', 
        # ... (resto de emojis)
    ]
    random.shuffle(agent_emoji)

    results = []
    for no, sample in enumerate(tqdm(test_qa)):
        if no == args.num_samples:
            break

        print(f"\n[INFO] no: {no}")
        question, img_path = create_question(sample, args.dataset)
        difficulty = determine_difficulty(question, args.difficulty)

        print(f"difficulty: {difficulty}")

        if difficulty == 'basic':
            final_decision = process_basic_query(question, examplers, args.model, args, img_path=img_path)
        elif difficulty == 'intermediate':
            final_decision = process_intermediate_query(question, examplers, args.model, args, img_path=img_path)
        elif difficulty == 'advanced':
            final_decision = process_advanced_query(question, args.model, args, img_path=img_path)

        # Para mycsvdataset, guardamos también los resultados
        results.append({
            'question': question,
            'label': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'response': final_decision,
            'difficulty': difficulty
        })

    # Guardar resultados
    path = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(path):
        os.makedirs(path)

    output_file = f'output/{args.model}_{args.dataset}_{args.difficulty}.json'
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"[INFO] Results saved at {output_file}")

if __name__ == "__main__":
    main()
