import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from starlette import status
from log import Logger
from fastapi.responses import JSONResponse

from npc.src.npc.crew import Npc

log = Logger(name="api-service")

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator:
    log.fire.info(f'{log.name} started successfully')
    yield
    log.fire.info(f'{log.name} finished successfully')


app = FastAPI()

@app.post('/talk')
async def talk(query: str) -> JSONResponse:
    npc = Npc()
    return JSONResponse(status_code=status.HTTP_200_OK, content={'response': npc.crew().kickoff(inputes={'query': query})})


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5555)
