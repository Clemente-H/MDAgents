import random
from src.agents import Agent

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append(f"({k}) {v}")
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    elif dataset == 'mycsvdataset':
        # Ya la pregunta tiene las opciones integradas, así que simplemente retornamos
        # la pregunta tal cual y la ruta de la imagen
        question = sample['question']
        img_path = sample['img_path']
        return question, img_path
    else:
        return sample['question'], None


def process_basic_query(question, examplers, model, args, img_path=None):
    # Agent para reason
    reasoning_agent = Agent('You are a helpful medical agent.', 'medical expert', model_info=model)
    new_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            tmp_exampler = {}
            exq = exampler['question']
            choices = [f"({k}) {v}" for k,v in exampler['options'].items()]
            random.shuffle(choices)
            exq += " " + ' '.join(choices)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            exampler_reason = reasoning_agent.chat(
                f"Below is a question and answer. Provide a short reasoning.\nQuestion: {exq}\n{exampler_answer}"
            )

            tmp_exampler['question'] = exq
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            new_examplers.append(tmp_exampler)
    
    single_agent = Agent(
        instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.',
        role='medical expert',
        examplers=new_examplers,
        model_info=model
    )
    single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
    final_decision = single_agent.temp_responses(
        f"The following are multiple choice questions.\n\nQuestion: {question}\nAnswer: ",
        img_path=img_path
    )
    return final_decision


def process_intermediate_query(question, examplers, model, args, img_path=None):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"You are an experienced medical expert who recruits a group of experts..."

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-3.5')
    tmp_agent.chat(recruit_prompt)

    num_agents = 5
    recruited = tmp_agent.chat(
        f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise..."
    )

    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

    agent_emoji = [
        '\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', 
        '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F',
        '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', 
        '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F',
        '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F'
    ]
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except:
            continue

        inst_prompt = f"You are a {agent_role} who {description}. Your job is to collaborate..."
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
        _agent.chat(inst_prompt, img_path=img_path)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    # Preparar ejemplares
    fewshot_examplers = ""
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        reasoning_agent = Agent('You are a helpful medical agent.', 'medical expert', model_info=model)
        for ie, exampler in enumerate(examplers[:5]):
            exq = exampler['question']
            options = [f"({k}) {v}" for k,v in exampler['options'].items()]
            random.shuffle(options)
            exq += " " + ' '.join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = tmp_agent.chat(
                f"Below is a question and answer. Provide reasoning.\n\nQuestion: {exq}\n\n{exampler_answer}",
                img_path=img_path
            )
            exq += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exq

    # Interacción y debate (resumido)
    # Suponemos N rondas y turns, ver código original para detalles.
    round_opinions = {1: {}}
    for k, v in agent_dict.items():
        opinion = v.chat(
            f"Given the exemplars, answer the question.\n\n{fewshot_examplers}\nQuestion: {question}\nAnswer: ",
            img_path=img_path
        )
        round_opinions[1][k.lower()] = opinion

    # Resumen y síntesis por un asistente
    agent_rs = Agent(
        instruction="You are a medical assistant who excels at summarizing...",
        role="medical assistant",
        model_info=model
    )
    agent_rs.chat("You are a medical assistant...", img_path=img_path)

    assessment = "".join(f"({k.lower()}): {v}\n" for k,v in round_opinions[1].items())
    report = agent_rs.chat(
        f"Here are some reports:\n{assessment}\n\nNow summarize.",
        img_path=img_path
    )

    # Determinar respuesta final (moderador)
    moderator = Agent(
        "You are a final medical decision maker who reviews all opinions...",
        "Moderator",
        model_info=model
    )
    moderator.chat("You are a final medical decision maker...", img_path=img_path)

    final_decision = moderator.temp_responses(
        f"Given each agent's final answer, review and make a final answer.\n{round_opinions[1]}\n\nQuestion: {question}",
        img_path=img_path
    )

    return final_decision

def process_advanced_query(question, model, args, img_path=None):
    # Reclutamiento de múltiples equipos (MDT), ver código original.
    recruit_prompt = "You are an experienced medical expert. Given the complex medical query..."
    tmp_agent = Agent(instruction=recruit_prompt, role='decision maker', model_info=model)
    tmp_agent.chat(recruit_prompt, img_path=img_path)

    num_teams = 3
    num_agents = 3
    recruited = tmp_agent.chat(
        f"Question: {question}\n\nOrganize {num_teams} MDTs with {num_agents} clinicians each...",
        img_path=img_path
    )

    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]

    group_instances = []
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        group_instance = Group(res_gs['group_goal'], res_gs['members'], question)
        group_instances.append(group_instance)

    # Interacciones iniciales de grupos, etc.
    # Por simplicidad, asumimos un flujo similar:
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower():
            init_assessment = group_instance.interact(comm_type='internal', img_path=img_path)
            initial_assessments.append([group_instance.goal, init_assessment])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"

    # Otros MDTs
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal', img_path=img_path)
            assessments.append([group_instance.goal, assessment])
    
    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    
    # Equipo final para decisión
    final_decisions = []
    for group_instance in group_instances:
        if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower():
            decision = group_instance.interact(comm_type='internal', img_path=img_path)
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"

    # Decisión final por un agente principal
    decision_prompt = "You are an experienced medical expert. Given the investigations from MDT..."
    decision_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model)
    decision_agent.chat(decision_prompt, img_path=img_path)
    
    final_decision = decision_agent.temp_responses(
        f"Investigation:\n{initial_assessment_report}\n\nQuestion: {question}",
        img_path=img_path
    )

    return final_decision