from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
from models.flower_classifier import FlowerClassifier
import shutil
from pathlib import Path
import uuid
import traceback

app = FastAPI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize the model with error handling
MODEL_PATH = r"C:\Users\vlad\education\practice_proj\jest_practice_site\src\flowers_efficientv2_acc_0.97.pth"

try:
    classifier = FlowerClassifier(MODEL_PATH)
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    classifier = None

# Mount static files
app.mount("/site", StaticFiles(directory="site"), name="site")

@app.get("/")
async def read_root():
    return FileResponse("site/index.html")

@app.get("/{path:path}")
async def read_static(path: str):
    file_path = Path("site") / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/classify")
async def classify_flower(file: UploadFile = File(...)):
    # Проверка, что модель загружена
    if classifier is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Модель классификации не загружена"}
        )
    
    # Проверка типа файла
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "Загруженный файл должен быть изображением"}
        )
    
    # Генерируем уникальное имя файла для избежания конфликтов
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = UPLOAD_DIR / unique_filename
    
    try:
        # Сохраняем файл
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Проверяем, что файл действительно сохранился
        if not temp_path.exists():
            return JSONResponse(
                status_code=500,
                content={"error": "Не удалось сохранить загруженный файл"}
            )
        
        # Получаем предсказания
        results = classifier.predict(str(temp_path))
        
        # Проверяем формат результата
        if not isinstance(results, list):
            return JSONResponse(
                status_code=500,
                content={"error": "Неверный формат ответа от модели"}
            )
        
        return results
        
    except Exception as e:
        print(f"Ошибка при классификации: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка при классификации: {str(e)}"}
        )
    finally:
        # Удаляем временный файл
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            print(f"Не удалось удалить временный файл: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)