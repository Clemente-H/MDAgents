from src.agents import Agent

def determine_difficulty(question, difficulty):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""Now, given the medical query as below, decide difficulty/complexity:
{question}\n
1) basic
2) intermediate
3) advanced
"""
    
    medical_agent = Agent(instruction='You are a medical expert who classifies complexity.', role='medical expert', model_info='gpt-3.5')
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'
    return 'basic'
