import os
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from .utils import encode_image
load_dotenv()

class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path:str = None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        self.examplers = examplers

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.getenv('openai_api_key'))
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            # Lógica para Gemini
            for _ in range(10):
                try:
                    # Si hay imagen, incluirla en el mensaje (dependerá de la API de Gemini)
                    if img_path:
                        response = self._chat.send_message([message, img_path], stream=True)
                    else:
                        response = self._chat.send_message(message, stream=True)
                    
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except Exception as e:
                    print(f"Error en Gemini: {e}")
                    continue
            return "Error: No se pudo obtener respuesta de Gemini."
        
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            # Preparar el contenido del mensaje
            if img_path:
                try:
                    # Codificar la imagen
                    image_b64 = encode_image(img_path)
                    
                    # Crear un mensaje con imagen
                    messages = self.messages + [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }]
                except Exception as e:
                    print(f"Error procesando imagen: {e}")
                    return "Error: No se pudo procesar la imagen"
            else:
                # Si no hay imagen, usar los mensajes existentes
                messages = self.messages + [{"role": "user", "content": message}]
            
            # Seleccionar el modelo correcto
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            # Realizar la solicitud a la API
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )

                # Obtener y guardar la respuesta
                response_content = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": response_content})
                return response_content
            
            except Exception as e:
                print(f"Error en la solicitud a OpenAI: {e}")
                return "Error: No se pudo obtener respuesta del modelo"

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            # Preparar el contenido del mensaje
            if img_path:
                try:
                    # Codificar la imagen
                    image_b64 = encode_image(img_path)
                    
                    # Crear contenido del mensaje con imagen
                    messages = self.messages + [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }]
                except Exception as e:
                    print(f"Error processing image: {e}")
                    return {"error": "Could not process image"}
            else:
                # Si no hay imagen, usar los mensajes existentes
                messages = self.messages + [{"role": "user", "content": message}]
                
            temperatures = [0.0]
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=messages,
                    temperature=temperature,
                )
                responses[temperature] = response.choices[0].message.content
            
            return responses
        
        elif self.model_info == 'gemini-pro':
            # Lógica para Gemini (aún no modificada para imagen)
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses