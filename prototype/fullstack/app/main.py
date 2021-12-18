from fastapi import FastAPI
import uvicorn
from predict import get_pipeline
from load_data import *

app = FastAPI()
pipe = get_pipeline()
dataset = load_data()

@app.get('/')
def hello_world():
    return {'hello': 'world'}


@app.get('/inference/{keyword}')
def inference(keyword):
    results = []
    comments = retrieve_comments(keyword, dataset)
    
    for comment in comments[:10]:
        output = pipe(comment)
    
        if output[0]['label'] == 'none':
            continue
    
        else:
            results.append(
                {
                    'comment': comment,
                    'label': output[0]['label'],
                    'score': output[0]['score']
                }
            )
    
    return results


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)