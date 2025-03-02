import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from inference import get_random_sample, preprocess_sample, inference, post_processing, execute_query

app = FastAPI()

# Get the parent directory (project root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "frontend")), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(ROOT_DIR, "frontend/index.html"), "r") as file:
        return HTMLResponse(content=file.read())

@app.post("/get_random_sample")
async def get_sample():
    sample = get_random_sample()
    return JSONResponse(content=sample)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    question = data.get("question")
    table = data.get("table")
    
    preprocessed_sample = preprocess_sample(question, table)
    model_output = inference(preprocessed_sample)
    print("Model Output:", model_output)  # Print model output for debugging
    pred_canonical, _ = post_processing(model_output, table["header"])
    print("Pred Canonical:", pred_canonical)  # Print pred_canonical for debugging
    
    response_content = {
        "query": pred_canonical,
        "model_output": model_output
    }
    print("Response Content:", response_content)  # Print the response content
    return JSONResponse(content=response_content)

@app.post("/execute_query")
async def execute(request: Request):
    data = await request.json()
    table_id = data.get("table_id")
    query = data.get("query")
    
    result = execute_query(table_id, query)
    return JSONResponse(content={"result": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7845)