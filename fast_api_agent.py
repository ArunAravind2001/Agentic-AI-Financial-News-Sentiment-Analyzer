
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# 👇 Import your LangGraph agent
from first_agentic_ai import graph  

app = FastAPI()


templates = Jinja2Templates(directory="templates")


class InputData(BaseModel):
    topic: str  


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze-news")
async def analyze_news(input_data: InputData):
   
    result = graph.invoke({"query": input_data.topic})

    return {
        "query": input_data.topic,
        "top_companies": result.get("top_companies", []),
        "articles": result["df"].to_dict(orient="records")  # convert DataFrame to list of dicts
    }


if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000) 
