import os
from openai import OpenAI
import google.generativeai as genai

class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        self.examplers = examplers

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.environ['openai_api_key'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.0]
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                responses[temperature] = response.choices[0].message.content
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses
