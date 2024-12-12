from pptree import Node
from typing import Union
from pathlib import Path
import base64
import pandas as pd
import os
import json
from datetime import datetime

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
        if line.startswith('- **Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info


def check_image_size(image_path, max_size_mb=5):
    # Convertimos MB a bytes (5MB = 5 * 1024 * 1024 bytes)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Obtenemos el tamaño del archivo
    file_size = os.path.getsize(image_path)
    
    return file_size <= max_size_bytes, file_size


def encode_image(image_path:str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
