import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executer_multi import HelloAgentExecutor


if __name__ == '__main__':
    skill = AgentSkill(
        id='hello_magentic_agent_multi',
        name='Hello Magentic Agent Multi-Session',
        description='A multi-session agent for managing Hello operations and use hello commands',
        tags=['hello', 'magentic', 'agent', 'multi-session'],
        examples=['say hello', 'greet user', 'ask for name'],
    )

    public_agent_card = AgentCard(
        name='Hello Magentic Agent Multi-Session',
        description='A multi-session agent for Hello related operations',
        url='http://localhost:9999/',
        version='2.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=HelloAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    print("Starting A2A Multi-Session Server on localhost:9999")
    uvicorn.run(server.build(), host='localhost', port=9999)
