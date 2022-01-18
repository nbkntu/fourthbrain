from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

app = FastAPI()

app.mount("/css", StaticFiles(directory="static/css"), name="static-css")
app.mount("/scripts", StaticFiles(directory="static/scripts"), name="static-scripts")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class HelloRequest(BaseModel):
    name: str

@app.post("/hello")
async def sayHello(hello_request: HelloRequest):
    return {"message": f"Hello, {hello_request.name}!"}
