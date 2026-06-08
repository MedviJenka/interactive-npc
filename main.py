import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from log import Logger


log = Logger(name="api-service")

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator:
    log.fire.info(f'{log.name} started successfully')
    yield
    log.fire.info(f'{log.name} finished successfully')


app = FastAPI()


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5555)