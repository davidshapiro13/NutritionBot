from llmproxy import LLMProxy

class AI():

    def __init__(self, model_name='gpt-5-mini', query_num=15):
        self.client = LLMProxy()
        self.model_name = model_name
        self.last_queries = query_num
        self.rag_enabled = False

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