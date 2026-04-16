from llmproxy import LLMProxy

class AI():

    def __init__(self, model_name='gpt-5-mini', query_num=15):
        self.client = LLMProxy()
        self.model_name = model_name
        self.last_queries = query_num
        self.rag_enabled = False

    def ask(self, system_prompt, query_prompt, session, media=None, model_override=None, lastk_override=None):
        output = self.client.generate(
            model = model_override or self.model_name,
            system = system_prompt,
            query = query_prompt,
            lastk = lastk_override if lastk_override is not None else self.last_queries,
            session_id = session,
            rag_usage = self.rag_enabled,
            rag_threshold = 0.5,
            media = media,
        )['result']
        return output
