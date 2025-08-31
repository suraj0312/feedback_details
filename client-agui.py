import os
import uuid
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
)
from ag_ui.encoder import EventEncoder

import httpx
import logging
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import AgentCard, MessageSendParams, SendStreamingMessageRequest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


app = FastAPI(title="AG-UI A2A Server")

# -------------------------------
# Initialize A2A Client globally
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_url = "http://localhost:9999"
httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(120))
a2a_client: A2AClient | None = None


async def init_a2a_client():
    global a2a_client
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
    try:
        logger.info(f"Fetching public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}")
        public_card: AgentCard = await resolver.get_agent_card()
        logger.info("Successfully fetched public agent card.")
        a2a_client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
        logger.info("A2AClient initialized.")
    except Exception as e:
        logger.error(f"Failed to fetch AgentCard: {e}", exc_info=True)
        raise


@app.on_event("startup")
async def startup_event():
    await init_a2a_client()


# -------------------------------
# FastAPI Endpoint
# -------------------------------
@app.post("/")
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    """AG-UI endpoint backed by A2AClient instead of OpenAI"""
    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def event_generator():
        try:
            # Run started
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

            # Prepare message payload (take the last user message from input_data)
            user_message = input_data.messages[-1]
            send_message_payload = {
                "message": {
                    "role": user_message.role,
                    "parts": [{"kind": "text", "text": user_message.content}],
                    "messageId": str(uuid.uuid4()),
                }
            }

            streaming_request = SendStreamingMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(**send_message_payload),
            )

            message_id = str(uuid.uuid4())

            # Stream response from A2A
            async for chunk in a2a_client.send_message_streaming(streaming_request):
                dumped = chunk.model_dump(mode="json", exclude_none=True)

                # Map A2A chunk to AG-UI event types
                if "message" in dumped and "parts" in dumped["message"]:
                    for part in dumped["message"]["parts"]:
                        if part.get("kind") == "text":
                            yield encoder.encode({
                                "type": EventType.TEXT_MESSAGE_CHUNK,
                                "message_id": message_id,
                                "delta": part["text"],
                            })

            # Run finished
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

        except Exception as error:
            yield encoder.encode(
                RunErrorEvent(type=EventType.RUN_ERROR, message=str(error))
            )

    return StreamingResponse(event_generator(), media_type=encoder.get_content_type())


def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("client-agui:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
