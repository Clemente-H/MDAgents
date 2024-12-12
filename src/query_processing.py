import random
from src.agents import Agent
from src.group import Group
from src.utils import parse_group_info

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
    single_agent = Agent(
        instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.',
        role='medical expert',
        examplers=[],
        model_info=model,
    )
    single_agent.chat(
        'You are a helpful assistant that answers multiple choice questions about medical knowledge.', 
        img_path)
    final_decision = single_agent.temp_responses(
        f"The following are multiple choice questions.\n\nQuestion: {question}\nAnswer: ",
        img_path=img_path
    )
    return final_decision


def process_intermediate_query(question, examplers, model, args, img_path=None):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"You are an experienced medical expert who recruits a group of experts..."

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-4o-mini')
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
    print("[STEP 1] Recruitment")
    group_instances = []
    # Reclutamiento de múltiples equipos (MDT), ver código original.
    recruit_prompt = f"""You are an experienced medical expert. 
    Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to
     make accurate and robust answer."""
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model)
    tmp_agent.chat(recruit_prompt, img_path=img_path)

    num_teams = 3
    num_agents = 3
    recruited = tmp_agent.chat(
        f"""Question: {question}
        
        You should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. 
        Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.


        For example, the following can an example answer:
        Group 1 - Initial Assessment Team (IAT)
        Member 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.
        Member 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.
        Member 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.
        
        Group 2 - Diagnostic Evidence Team (DET)
        Member 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.
        Member 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.
        Member 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.
        
        Group 3 - Patient History Team (PHT)
        Member 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.
        Member 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.
        Member 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.
        
        Group 4 - Final Review and Decision Team (FRDT)
        Member 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision
        Member 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.
        Member 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.
        
        Above is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.""",img_path)


    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]

    #fix first empty group, wronged parsed
    group_strings.pop(0)
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")

        group_instance = Group(res_gs['group_goal'], res_gs['members'], question, examplers=None, img_path=img_path)
        group_instances.append(group_instance)

    print("[STEP 2] Initial assessment from each group")
    # Interacciones iniciales de grupos, etc.
    # Por simplicidad, asumimos un flujo similar:
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
            init_assessment = group_instance.interact(comm_type='internal',message=None, img_path=img_path)
            initial_assessments.append([group_instance.goal, init_assessment])
    
    initial_assessment_report = ""
    print("Initial Assessment", init_assessment)
    print("Reports:")
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"
        print(idx)
        print(initial_assessment_report)
    # Otros MDTs
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal', img_path=img_path)
            assessments.append([group_instance.goal, assessment])
    print("otros Assessments", assessments)

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