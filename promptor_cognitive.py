from IPython import embed
import re
    
class ParaphrasedPrompter:
    def __init__(self) -> None:
        self.demo_instruction = "Introduction: Your task is to paraphrase the provided conversation while preserving the original intent and meaning. Each turn in the conversation, including queries and responses, should be paraphrased thoughtfully.\nExample to Illustrate the Process:\n"
        self.instruction = "Now, it's your turn. Please paraphrase the following conversation using the same process:\n"
        self.stop_tokens = ['\n']
        self.demo_sep = "\n\n\n"
    
    def make_demo_prompt(self, demo):
        original_conv = demo['original']
        paraphrased_conv = demo['paraphrased']
        cognitive_mapping = demo['cognitive_mapping']
        asso_div = demo['asso_div']

        prompt = ""

        prompt += "Original Conversation:\n"
        for i in range(len(original_conv)):
            prompt += f"Query{original_conv[i]['turn_id']}: {original_conv[i]['question']}\t"
            if i<len(original_conv)-1:
                prompt += f"Response{original_conv[i]['turn_id']}: {original_conv[i]['response']}\t"
                
        prompt += "\nStep 1: Comprehension Synthesis (Identify key themes and intents)\n"
        prompt += cognitive_mapping + '\n'
        
        prompt += "Step 2: Associative Expansion  (Generate alternative expressions)\n"
        prompt += asso_div + '\n'
        
        prompt += "Step 3: Conclusion (Reconstruct the conversation)\n"
        prompt += "\nParaphrased Conversation:\n"
        for i in range(len(paraphrased_conv)):
            prompt += f"Query{paraphrased_conv[i]['turn_id']}: {paraphrased_conv[i]['question']}\t"
            if i<len(original_conv)-1:
                prompt += f"Response{paraphrased_conv[i]['turn_id']}: {paraphrased_conv[i]['response']}\t"

        return prompt


    def make_query_prompt(self, turn_id, question):
        return f"Query{turn_id}: {question}\t"
    
    def make_response_prompt(self, turn_id, response):
        return f"Response{turn_id}: {response}\t"

    def parse_returned_text(self, text):
        return text.strip()

class ExtendPrompter:
    def __init__(self) -> None:
        self.demo_instruction = "Your task is to introduce a noisy turn (one query and one response) into an existing conversation. This turn should be relevant to the main background of the original conversation but introduce a new, slightly divergent element. Use the following structured approach:\n"
        self.instruction = "Now, it's your turn. Please introduce a noisy turn into the following conversation using the same process:\n"
        self.stop_tokens = ['\n']
        self.demo_sep = "\n\n\n"
    
    def make_demo_prompt(self, demo):
        original_conv = demo['original']
        additional_turns = demo['additional_turns']
        cognitive_mapping = demo['cognitive_mapping']
        cont_diver = demo['cont_diver']

        prompt = ""

        prompt += "Original Conversation:\n"
        for i in range(len(original_conv)):
            prompt += f"Query{original_conv[i]['turn_id']}: {original_conv[i]['question']}\t"
            # if i<len(original_conv)-1:
            prompt += f"Response{original_conv[i]['turn_id']}: {original_conv[i]['response']}\t"
                
        prompt += "\nStep 1: Comprehension Synthesis (Identify key themes and intents)\n"
        prompt += cognitive_mapping + '\n'
        
        prompt += "Step 2: Associative Expansion (Generate a related but distinct element)\n"
        prompt += cont_diver + '\n'
        
        prompt += "Step 3: Conclusion (Introduce the new turn)\n"
        prompt += "Noisy Turn:\n"
        for i in range(len(additional_turns)):
            prompt += f"Query{additional_turns[i]['turn_id']}: {additional_turns[i]['question']}\t"
            # if i<len(original_conv)-1:
            prompt += f"Response{additional_turns[i]['turn_id']}: {additional_turns[i]['response']}\t"

        return prompt


    def make_query_prompt(self, turn_id, question):
        return f"Query{turn_id}: {question}\t"
    
    def make_response_prompt(self, turn_id, response):
        return f"Response{turn_id}: {response}\t"

    def parse_returned_text(self, text):
        return text.strip()

class EntityPrompter:
    def __init__(self) -> None:
        self.demo_instruction = "Introduction: Your task is to replace entities in the current conversation context while keeping the expressions as similar as possible to the original. This involves identifying key entities, replacing them with suitable alternatives, and ensuring the conversation remains coherent. Use the following structured approach:\n"
        self.instruction = "Now, it's your turn. Please replace entities in the following conversation using the same process:\n"
        self.stop_tokens = ['\n']
        self.demo_sep = "\n\n\n"
    
    def make_demo_prompt(self, demo):
        original_conv = demo['original']
        entity_replaced = demo['entity_replaced']
        entity = demo['entity']
        substitution = demo['substitution']

        prompt = ""

        prompt += "Original Conversation:\n"
        for i in range(len(original_conv)):
            prompt += f"Query{original_conv[i]['turn_id']}: {original_conv[i]['question']}\t"
            if i<len(original_conv)-1:
                prompt += f"Response{original_conv[i]['turn_id']}: {original_conv[i]['response']}\t"
                
        prompt += "\nStep 1: Comprehension Synthesis (Identify key entities in the conversation)\n"
        prompt += entity + '\n'
        
        prompt += "Step 2: Associative Expansion (Find suitable replacements for the identified entities)\n"
        prompt += substitution + '\n'
        
        prompt += "Step 3: Conclusion (Reconstruct the conversation with new entities)\n"
        prompt += "\nSwapped Conversation:\n"
        for i in range(len(entity_replaced)):
            prompt += f"Query{entity_replaced[i]['turn_id']}: {entity_replaced[i]['question']}\t"
            if i<len(original_conv)-1:
                prompt += f"Response{entity_replaced[i]['turn_id']}: {entity_replaced[i]['response']}\t"

        return prompt


    def make_query_prompt(self, turn_id, question):
        return f"Query{turn_id}: {question}\t"
    
    def make_response_prompt(self, turn_id, response):
        return f"Response{turn_id}: {response}\t"

    def parse_returned_text(self, text):
        return text.strip()


class ShiftPrompter:
    def __init__(self) -> None:
        self.demo_instruction = "Introduction: Your task is to modify the current conversation by shifting its search intent. The new conversation should retain similar expressions to the original but embody a distinctly different intent. Follow this structured approach:\n"
        self.instruction = "Now, it's your turn. Please shift the intent of the following conversation using the same process:\n"
        self.stop_tokens = ['\n']
        self.demo_sep = "\n\n\n"
    
    def make_demo_prompt(self, demo):
        original_conv = demo['original']
        entity_replaced = demo['shifted']
        cognitive_mapping = demo['cognitive_mapping']
        new_intent = demo['new_intent']

        prompt = ""

        prompt += "Original Conversation:\n"
        for i in range(len(original_conv)):
            prompt += f"Query{original_conv[i]['turn_id']}: {original_conv[i]['question']}\t"
            if i<len(original_conv)-1:
                prompt += f"Response{original_conv[i]['turn_id']}: {original_conv[i]['response']}\t"
                
        prompt += "\nStep 1: Comprehension Synthesis (Identify key themes and intents)\n"
        prompt += cognitive_mapping + '\n'
        
        prompt += "Step 2: Associative Expansion (Choose a distinctly different intent)\n"
        prompt += new_intent + '\n'
        
        prompt += "Step 3: Conclusion (Reconstruct the conversation with the new intent)\n"
        prompt += "\nIntent-Shifted Conversation:\n"
        for i in range(len(entity_replaced)):
            prompt += f"Query{entity_replaced[i]['turn_id']}: {entity_replaced[i]['question']}\t"
            if i<len(original_conv)-1:
                prompt += f"Response{entity_replaced[i]['turn_id']}: {entity_replaced[i]['response']}\t"

        return prompt


    def make_query_prompt(self, turn_id, question):
        return f"Query{turn_id}: {question}\t"
    
    def make_response_prompt(self, turn_id, response):
        return f"Response{turn_id}: {response}\t"

    def parse_returned_text(self, text):
        return text.strip()

class DependencyPrompter:
    def __init__(self) -> None:
        self.demo_instruction = "Introduction: Your task is to analyze a given conversation and identify the turns that are necessary for understanding the current search intent. Each turn includes one query and one response. Follow this structured approach:"
        self.instruction = "Now, it's your turn. Please identify the necessary turns in the following conversation using the same process:"
        # self.instruction = "Now, it's your turn. Please analyze the following conversation context and identify turns that are necessary for interpreting the given search intent. You should only output a list of the necessary turns' names and use a comma to separate them (Like Turn x,Turn m,...). If None turns in the context are necessary, output 'None'. Always output a minimal set of turns that is sufficient to infer the current search intent:"
        self.stop_tokens = ['\n']
        self.demo_sep = "\n\n\n"
    
    def make_demo_prompt(self, demo):
        original_conv = demo['original']
        cognitive_mapping = demo['cognitive_mapping']
        rele_assess = demo['rele_assess']
        dependency = demo['dependency']

        prompt = ""

        
        prompt += "\nConversation Context:\n"
        for i in range(len(original_conv)):
            if i<len(original_conv)-1:
                prompt += f"Turn{i+1}: Query{original_conv[i]['turn_id']}: {original_conv[i]['question']}\tResponse{original_conv[i]['turn_id']}: {original_conv[i]['response']}\t"
            else:
                prompt += f"Query{original_conv[i]['turn_id']}: {original_conv[i]['question']}\t"

        prompt += "Step 1: Comprehension Synthesis (Identify key themes and intents)\n"
        prompt += cognitive_mapping + '\n'
        
        prompt += "Step 2: Associative Expansion (Evaluate the importance of each turn in relation to the current intent)\n"
        prompt += rele_assess + '\n'
        
        prompt += "Step 3: Conclusion (Select turns crucial for the current intent)\n"
        prompt += "\nNecessary Turns:\n"
        prompt += f"{dependency}"
        
        
        return prompt


    def make_query_prompt(self, turn_id, question):
        return f"Query{turn_id}: {question}\t"
    
    def make_turn_prompt(self, turn_id, question, response):
        return f"Turn{turn_id}: Query{turn_id}: {question}\tResponse{turn_id}: {response}\t"
    
    def make_oquery_prompt(self, question):
        return f"{question}"
    
    def make_response_prompt(self, turn_id, response):
        return f"Response{turn_id}: {response}\t"

    def parse_returned_text(self, text):
        return text.strip()


class PPLPrompter:
    def __init__(self) -> None:
        self.demo_instruction = "I'm going to provide you with a conversation context and a current query. Your task is to answer the current query based on the information of the context:\n\n"
        self.instruction = "Now, it's your turn. Please answer the following conversation:\n\n"
        self.stop_tokens = ['\n']
        self.demo_sep = "\n\n\n"
    
    def make_demo_prompt(self, demo):
        original_conv = demo['original']

        prompt = ""

        prompt += "Conversation Context:\n"
        for i in range(len(original_conv)-1):
            prompt += f"Query{original_conv[i]['turn_id']}: {original_conv[i]['question']}\t"
            prompt += f"Response{original_conv[i]['turn_id']}: {original_conv[i]['response']}\t"

        i=len(original_conv)-1
        
        prompt += "\nCurrent Query:\n"
        prompt += f"{original_conv[i]['question']}\t"
        
        prompt += "Response:\n"
        prompt += f"{original_conv[i]['response']}\t"
        return prompt


    def make_query_prompt(self, turn_id, question):
        return f"Query{turn_id}: {question}\t"
    
    def make_response_prompt(self, turn_id, response):
        return f"Response{turn_id}: {response}\t"
    
    def make_only_query_prompt(self, turn_id, question):
        return f"Query: {question}\t"
    
    def make_only_response_prompt(self, turn_id, response):
        return f"Response: {response}\t"

    def parse_returned_text(self, text):
        return text.strip()