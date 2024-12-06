from src.agents import Agent

class Group:
    def __init__(self, goal, members, question, examplers=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent(
                f'You are a {member_info["role"]} who {member_info["expertise_description"].lower()}.',
                role=member_info['role'],
                model_info='gpt-4o-mini'
            )
            _agent.chat(f'You are a {member_info["role"]} who {member_info["expertise_description"].lower()}.')
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role
                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]

            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += f"\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {self.question}"
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat(f"You are in a medical group where the goal is to {self.goal}. Your group lead is asking for the following investigations:\n{delivery}\n\nPlease remind your expertise and return your investigation summary.")
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += f"[{investigation[0]}]\n{investigation[1]}\n"

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your assistant clinicians is as follows:\n{gathered_investigation}\n\nAfter reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your assistant clinicians is as follows:\n{gathered_investigation}\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)
            return response

        elif comm_type == 'external':
            # Lógica si es necesaria
            pass
