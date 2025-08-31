from autogen_agentchat.base import TaskResult
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from agent import GitMagenticAgent, model_client
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    # Task,
    TaskState,
    TextPart,

)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)



class GitMagenticAgentExecutor(AgentExecutor):
    """GitAgent implementation."""

    def __init__(self):
        self.agent = GitMagenticAgent()
        


    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task
   
        if not task:
            if not context.message:
                raise ValueError("Context message is missing or invalid.")
            task = new_task(context.message)
      
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for message in self.agent.team.run_stream(task=query, output_task_messages=True):
                if not isinstance(message, TaskResult):
                    await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(str(message.content), task.context_id, task.id),
                    )
                    continue
               
                await updater.add_artifact(
    
                    [Part(root=TextPart(text=message.stop_reason))],
                    name='response',
                )
                await updater.complete()
                break
        except Exception as e:
            print(f"Error during agent execution: {e}")
        finally:
            self.agent.team.reset()
            model_client.close()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported')


