import os
import uuid
import logging
from typing import Dict
import uvicorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import AgentCard, MessageSendParams, SendStreamingMessageRequest
import httpx

# --- AG-UI event format ---
from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    TextMessageContentEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    CustomEvent
)
from ag_ui.encoder import EventEncoder
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AG-UI A2A Multi-Session Server")

origins = [
    "http://localhost:3000",  # Next.js dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

a2a_client: A2AClient | None = None
httpx_client: httpx.AsyncClient | None = None

# ---------------------------
# Multi-Session UserInputManager
# ---------------------------
class MultiSessionUserInputManager:
    def __init__(self):
        # Session-isolated input management
        self.session_input_events: Dict[str, Dict[str, asyncio.Event]] = {}
        self.session_input_values: Dict[str, Dict[str, str]] = {}

    def get_session_id_from_task_id(self, task_id: str) -> str:
        """Extract session info from task message or create new session."""
        # For this implementation, we'll use the task_id as session identifier
        # In production, you might want a more sophisticated session mapping
        return f"session_{task_id}"

    async def wait_for_input(self, task_id: str, session_id: str = None) -> str:
        if session_id is None:
            session_id = self.get_session_id_from_task_id(task_id)
        
        if session_id not in self.session_input_events:
            self.session_input_events[session_id] = {}
            self.session_input_values[session_id] = {}
        
        if task_id not in self.session_input_events[session_id]:
            self.session_input_events[session_id][task_id] = asyncio.Event()
            print(f"[Session {session_id}] Created input event for task {task_id}")
        
        print(f"[Session {session_id}] Waiting for input for task {task_id}")
        await self.session_input_events[session_id][task_id].wait()
        
        value = self.session_input_values[session_id].pop(task_id)
        del self.session_input_events[session_id][task_id]
        
        print(f"[Session {session_id}] Received input value: {value}")
        return value

    def provide_input(self, task_id: str, value: str, session_id: str = None):
        if session_id is None:
            session_id = self.get_session_id_from_task_id(task_id)
        
        if session_id not in self.session_input_values:
            self.session_input_values[session_id] = {}
            self.session_input_events[session_id] = {}
        
        self.session_input_values[session_id][task_id] = value
        print(f"[Session {session_id}] Stored input for task {task_id}: {value}")
        
        if task_id in self.session_input_events[session_id]:
            self.session_input_events[session_id][task_id].set()
            print(f"[Session {session_id}] Input event set for task {task_id}")

    def cleanup_session(self, session_id: str):
        """Clean up session data."""
        if session_id in self.session_input_events:
            del self.session_input_events[session_id]
        if session_id in self.session_input_values:
            del self.session_input_values[session_id]
        print(f"Cleaned up session: {session_id}")

# ✅ Multi-session singleton
multi_session_user_input_manager = MultiSessionUserInputManager()

@app.on_event("startup")
async def startup_event():
    """Initialize A2A client on startup"""
    global a2a_client
    global httpx_client

    base_url = "http://localhost:9999"
    logger.info(f"Connecting to A2A server at {base_url}")

    # Create global AsyncClient
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(120))

    # Resolve AgentCard
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
    try:
        card: AgentCard = await resolver.get_agent_card()
        logger.info("Fetched public agent card successfully.")
        a2a_client = A2AClient(httpx_client=httpx_client, agent_card=card)
    except Exception as e:
        logger.error(f"Failed to fetch agent card: {e}", exc_info=True)
        raise


@app.post("/")
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    """Multi-session A2A agentic chat endpoint"""
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)
    
    # Generate session ID from thread_id and run_id
    session_id = f"session_{input_data.thread_id}_{input_data.run_id}"

    async def event_generator():
        try:
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

            message_id = str(uuid.uuid4().hex)
            send_message_payload = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": input_data.messages[-1].content}],
                    "messageId": message_id,
                }
            }

            streaming_request = SendStreamingMessageRequest(
                id=str(uuid.uuid4().hex), params=MessageSendParams(**send_message_payload)
            )

            stream_response = a2a_client.send_message_streaming(streaming_request)

            first_chunk_text = True
            async for task_event in stream_response:
                dumped_event = task_event.model_dump(mode="json", exclude_none=True)
                
                if dumped_event["result"]["kind"] in ["task", "status-update"]:
                    status = dumped_event["result"]["status"]["state"]
                    print(f"[Session {session_id}] Status: {status}")

                    if status == "input-required":
                        task_id = dumped_event["result"]["status"]["message"]["taskId"]
                        print(f"[Session {session_id}] Input required for task: {task_id}")
                        
                        # Wait for session-specific input
                        user_value = await multi_session_user_input_manager.wait_for_input(task_id, session_id)

                        # Send follow-up message with session context
                        followup_payload = {
                            "message": {
                                "role": "user",
                                "parts": [{"kind": "text", "text": user_value}],
                                "contextId": dumped_event["result"]["contextId"],
                                "messageId": str(uuid.uuid4().hex),
                                "taskId": task_id
                            }
                        }
                        send_req = SendStreamingMessageRequest(
                            id=dumped_event["id"], 
                            params=MessageSendParams(**followup_payload)
                        )

                        try:
                            resp = a2a_client.send_message_streaming(send_req)
                            print(f"[Session {session_id}] Sent user input back to A2A server")
                            
                            async for event in resp:
                                event_json = event.model_dump(mode="json", exclude_none=True)
                                print(f"[Session {session_id}] Follow-up response: {event_json}")

                        except Exception as ex:
                            print(f"[Session {session_id}] Failed to send follow-up message: {ex}")

                        yield encoder.encode(
                            CustomEvent(
                                type=EventType.CUSTOM,
                                name="User Input",
                                value={
                                    "user_value": user_value, 
                                    "request_id": task_id,
                                    "session_id": session_id
                                }
                            )
                        )

                    elif status == "working" and "message" in dumped_event["result"]["status"]:
                        # Process streaming text content
                        message_content = dumped_event["result"]["status"]["message"]["parts"][0]["text"]
                        
                        if first_chunk_text:
                            yield encoder.encode(
                                TextMessageStartEvent(
                                    type=EventType.TEXT_MESSAGE_START,
                                    message_id=message_id,
                                )
                            )
                            first_chunk_text = False

                        yield encoder.encode(
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message_id,
                                delta=message_content,
                            )
                        )

                elif dumped_event["result"]["kind"] == "artifact-update":
                    artifact_name = dumped_event["result"]["artifact"]["name"]
                    if artifact_name == 'response':
                        text_parts = dumped_event["result"]["artifact"].get("parts", [])
                        final_response = text_parts[0]["text"]

                        if not first_chunk_text:
                            yield encoder.encode(
                                TextMessageEndEvent(
                                    type=EventType.TEXT_MESSAGE_END,
                                    message_id=message_id,
                                )
                            )
                        
                        yield encoder.encode(
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message_id,
                                delta=final_response,
                            )
                        )
                        
                        yield encoder.encode(
                            RunFinishedEvent(
                                type=EventType.RUN_FINISHED,
                                thread_id=input_data.thread_id,
                                run_id=input_data.run_id,
                            )
                        )
                        
                        # Clean up session after completion
                        multi_session_user_input_manager.cleanup_session(session_id)
                        return

            if not first_chunk_text:
                yield encoder.encode(
                    TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=message_id,
                    )
                )
            
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

        except Exception as error:
            logger.error(f"[Session {session_id}] Error in event_generator: {error}", exc_info=True)
            yield encoder.encode(
                RunErrorEvent(type=EventType.RUN_ERROR, message=str(error))
            )

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/input")
async def provide_input(request_id: str, value: str, session_id: str = None):
    """Provide input for specific session and task"""
    print(f"Received /input POST for request_id={request_id}, session_id={session_id}: {value}")
    multi_session_user_input_manager.provide_input(request_id, value, session_id)
    return JSONResponse({
        "status": "ok", 
        "request_id": request_id, 
        "value": value,
        "session_id": session_id
    })


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    print("Starting AG-UI Multi-Session Server on 0.0.0.0:8000")
    uvicorn.run("agui_server_multi:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
