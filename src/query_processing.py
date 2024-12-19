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
    print("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
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
    history_process = ""
    #print("[STEP 1] Recruitment")
    history_process += f"[PASO 1] Reclutamiento \n\n"
    group_instances = []
    # Reclutamiento de múltiples equipos (MDT), ver código original.
    recruit_prompt = f"""Eres un médico experto con mucha experiencia. 
    Dada la compleja consulta médica, necesitas organizar Equipos Multidisciplinarios (MDTs) y los miembros de cada MDT para
    proporcionar una respuesta precisa y robusta."""
    tmp_agent = Agent(
        instruction=recruit_prompt, 
        role='recruiter', 
        model_info=model)

    tmp_agent.chat(recruit_prompt, img_path=img_path)

    num_teams = 3
    num_agents = 3
    recruited = tmp_agent.chat(
        f"""Pregunta: {question}
        
        Debes organizar {num_teams} MDTs con diferentes especialidades o propósitos, y cada MDT debe tener {num_agents} clínicos. 
        Considerando la pregunta médica y las opciones, por favor devuelve tu plan de reclutamiento para ofrecer una respuesta más precisa.


        Por ejemplo, el siguiente puede ser un ejemplo de respuesta:
        Grupo 1 - Equipo de Evaluación Inicial (IAT)
        Miembro 1: Otorrinolaringólogo (Cirujano ENT) (Líder) - Especialista en cirugía de oído, nariz y garganta, incluyendo tiroidectomía. Este miembro lidera el grupo debido a su papel crítico en la intervención quirúrgica y la gestión de cualquier complicación quirúrgica, como el daño a los nervios.
        Miembro 2: Cirujano General - Proporciona experiencia quirúrgica adicional y apoyo en la gestión general de las complicaciones de la cirugía tiroidea.
        Miembro 3: Anestesiólogo - Se centra en el cuidado perioperatorio, el manejo del dolor y la evaluación de cualquier complicación de la anestesia que pueda afectar la voz y la función de las vías respiratorias.
        
        Grupo 2 - Equipo de Evidencia Diagnóstica (DET)
        Miembro 1: Endocrinólogo (Líder) - Supervisa la gestión a largo plazo de la enfermedad de Graves, incluyendo la terapia hormonal y el monitoreo de cualquier complicación relacionada después de la cirugía.
        Miembro 2: Patólogo del Habla y Lenguaje - Especialista en trastornos de la voz y deglución, proporcionando servicios de rehabilitación para mejorar la calidad del habla y la voz del paciente tras el daño nervioso.
        Miembro 3: Neurólogo - Evalúa y asesora sobre el daño nervioso y las posibles estrategias de recuperación, aportando experiencia neurológica al cuidado del paciente.
        
        Grupo 3 - Equipo de Historia del Paciente (PHT)
        Miembro 1: Psiquiatra o Psicólogo (Líder) - Aborda cualquier impacto psicológico de la enfermedad crónica y sus tratamientos, incluyendo problemas relacionados con los cambios en la voz, la autoestima y las estrategias de afrontamiento.
        Miembro 2: Fisioterapeuta - Ofrece ejercicios y estrategias para mantener la salud física y potencialmente apoyar la recuperación de la función vocal indirectamente a través del bienestar general.
        Miembro 3: Terapeuta Ocupacional - Ayuda al paciente a adaptarse a los cambios en la voz, especialmente si su profesión depende en gran medida de la comunicación vocal, ayudándolo a encontrar estrategias para mantener sus roles ocupacionales.
        
        Grupo 4 - Equipo de Revisión y Decisión Final (FRDT)
        Miembro 1: Consultor Senior de cada especialidad (Líder) - Proporciona experiencia general y orientación en la toma de decisiones
        Miembro 2: Especialista en Decisiones Clínicas - Coordina las diferentes recomendaciones de los diversos equipos y formula un plan de tratamiento integral.
        Miembro 3: Soporte Diagnóstico Avanzado - Utiliza herramientas y técnicas de diagnóstico avanzadas para confirmar el alcance exacto y la causa del daño nervioso, ayudando en la decisión final.
        
        Lo anterior es solo un ejemplo, por lo tanto, debes organizar tus propios MDTs únicos, pero debes incluir al Equipo de Evaluación Inicial (IAT) y al Equipo de Revisión y Decisión Final (FRDT) en tu plan de reclutamiento. Al devolver tu respuesta, refiérete estrictamente al formato anterior.""",
        img_path)


    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]

    #fix first empty group, wronged parsed
    group_strings.pop(0)
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
    #    print(f"Group {i1+1} - {res_gs['group_goal']}")
        history_process += f"Grupo {i1+1} - {res_gs['group_goal']}\n"
        for i2, member in enumerate(res_gs['members']):
    #        print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
            history_process += f" Miembro {i2+1} ({member['role']}): {member['expertise_description']}\n"

        group_instance = Group(
            res_gs['group_goal'], 
            res_gs['members'], 
            question, 
            examplers=None, 
            img_path=img_path)
        group_instances.append(group_instance)

    #print("[STEP 2] Initial assessment from each group")
    history_process += f"\n[PASO 2] Evaluación inicial de cada grupo\n\n"
    # Interacciones iniciales de grupos, etc.
    # Por simplicidad, asumimos un flujo similar:
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
            init_assessment = group_instance.interact(comm_type='internal',img_path=img_path)
            initial_assessments.append([group_instance.goal, init_assessment])
    #print("Initial Assessment", initial_assessments)

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Grupo {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"
    #print("Reports:", initial_assessment_report)
    history_process += initial_assessment_report
    # Otros MDTs STEP 2.2
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal', img_path=img_path)
            assessments.append([group_instance.goal, assessment])
    #print("otros Assessments", assessments)

    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Grupo {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    #print('Otros Assessment report: ', assessment_report)
    history_process += assessment_report

    #STEP 2.3 Equipo final para decisión
    final_decisions = []
    for group_instance in group_instances:
        if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower():
            decision = group_instance.interact(comm_type='internal', img_path=img_path)
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Grupo {idx+1} - {decision[0]}\n{decision[1]}\n\n"
    #print('Compiled report: ', compiled_report)
    history_process += compiled_report

    # Decisión final por un agente principal
    decision_prompt =f"""Eres un médico experto. Dadas las investigaciones de los equipos multidisciplinarios (MDT), por favor revísalas muy cuidadosamente y devuelve tu decisión final para la consulta médica.
    Siempre comienza con tu decisión final. Después de eso, proporciona la explicación. Recuerda siempre responder en español y siempre indicar la alternativa escogida."""
    decision_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model)
    decision_agent.chat(decision_prompt, img_path=img_path)
    
    # Para OpenAI podemos usar temp_responses, para otros proveedores usamos chat normal

    final_decision = decision_agent.temp_responses(
        f"Investigación:\n{initial_assessment_report}\n\nPregunta: {question}",
        img_path=img_path
    )
    history_process += str(final_decision)
    return final_decision, history_process