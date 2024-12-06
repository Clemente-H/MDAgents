from pptree import Node

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node(f"{child} ({emojis[count]})", agent)
                    agents.append(child_agent)

        else:
            agent = Node(f"{expert} ({emojis[count]})", moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info
