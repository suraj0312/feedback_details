import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executer import (
    GitMagenticAgentExecutor,  # type: ignore[import-untyped]
)



if __name__ == '__main__':
    # --8<-- [start:AgentSkill]
    skill = AgentSkill(
        id='git_magentic_agent',
        name='Git Magentic Agent',
        description='An agent for managing Git repositories and use git commands',
        tags=['git', 'magentic', 'agent'],
        examples=['clone a repo', 'create a branch', 'commit changes', 'push changes'],
    )


    public_agent_card = AgentCard(
    name='Git Magentic Agent',
    description='An agent for Git related operations',
    url='http://localhost:9999/',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],  # Only the basic skill for the public card
    supports_authenticated_extended_card=True,
    )


    request_handler = DefaultRequestHandler(
    agent_executor=GitMagenticAgentExecutor(),
    task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
        # extended_agent_card=specific_extended_agent_card,
    )

    uvicorn.run(server.build(), host='localhost', port=9999)