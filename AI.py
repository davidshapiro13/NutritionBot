from llmproxy import LLMProxy

class AI():

    def __init__(self, model_name='us.anthropic.claude-3-haiku-20240307-v1:0', query_num=10):
        self.client = LLMProxy()
        self.model_name = model_name
        self.last_queries = query_num
        self.rag_enabled = True

    def ask(self, system_prompt, query_prompt, session):
        output = self.client.generate(
            model = self.model_name,
            system = system_prompt,
            query = query_prompt,
            lastk = self.last_queries,
            session_id = session,
            rag_usage = self.rag_enabled,
            rag_threshold = 0.5
        )['result']
        return output
    
    