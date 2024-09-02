from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import google.generativeai as genai
import time, json

GOOGLE_API_KEY = 'AIzaSyAlkJtmJG3ScdI67adyDpwnxHiwrN4cf1M'
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0827",
    generation_config=generation_config,
)


@app.post("/process_video/")
async def process_video(video_file: UploadFile = File(...),
                        prompt: str = "from the video i gave get the facial and voice emotions via recognition .. also suggest some workout exercises based on physique and all parameters also suggest some nutrition and give in a very detailed json format .. keep the things brief .. also give the emotions in the format .. do give the remarks at the end if required else keep it NA"):
    try:
        print(f"Uploading file: {video_file.filename}...")
        genai_video_file=genai.upload_file(video_file.filename)

        print(f"Completed upload: {genai_video_file.uri}")

        timeout = 600
        start_time = time.time()
        while genai_video_file.state.name == "PROCESSING" and (time.time() - start_time) < timeout:
            print('.', end='')
            time.sleep(10)
            genai_video_file = genai.get_file(genai_video_file.name)

        if genai_video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {genai_video_file.state.name}")
        elif genai_video_file.state.name == "PROCESSING":
            raise TimeoutError("Video processing timed out")

        print("Making LLM inference request...")
        response = model.generate_content([prompt, genai_video_file],
                                          request_options={"timeout": 600})
        data = json.loads(response.text)

        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
