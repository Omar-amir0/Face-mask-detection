# شوية شرح من عمر عشان لو حد عايز يجربه

# Run command: uvicorn app:app --reload --port 8000
#then open google chrome and paste this: on google or oduble click it in the terminal
#http://127.0.0.1:8000/ 
#if u want to add files add /docs so it becomes 
#http://127.0.0.1:8000/docs then in green box press on post then try it oout 
#scroll down u will see it ask u to upload file put file and scroll down to see results (it takes mutliple photos i think)

from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import io
from PIL import Image
from torchvision import transforms, models


app = FastAPI(title="Face Mask Detection API")
# fastAPI بنحمل ال  

CLASSES = ['WithMask', 'WithoutMask']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device uses gpu

# نعمل احلي موديل بس منغير اوزان عشان بيخسس
model = models.mobilenet_v2(weights=None)  

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.last_channel, 2)
)
# هننسي اخر اتنين 

# نحيب الموديل الحبيب
checkpoint = torch.load("mask_detector.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)

model.eval()
# نقيمة

# شوية تضبيط فالاول كدا
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),      
    transforms.ToTensor(),            
    transforms.Normalize(               
        [0.485, 0.456, 0.406],          
        [0.229, 0.224, 0.225]               
    )
])
# جايبة من ملف الي يوسف عامله :)

@app.post("/predict")

async def predict(file: UploadFile = File(...)): #stolen from project requirements (معرفش ايه ديه)

    # صورك الجميلة
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    # شوية فلاتر عشان البشرة  :>
    tensor = preprocess(img).unsqueeze(0)

    
    with torch.no_grad(): #stolen from templates sheet
        
        probs = torch.softmax(model(tensor), dim=1)[0]
    
    pred_class_idx = probs.argmax().item()
    
    pred_class = CLASSES[pred_class_idx] #gets classes names with or without mask
    
    confidence = float(probs.max())
    
    status = "mask_on" if pred_class == "WithMask" else "mask_off"
    action = "Allow entry" if status == "mask_on" else "Deny entry"
    
    #
    return {
        "class": pred_class,         
        "confidence": confidence,       
        "status": status,             
        "action": action            
    }

@app.get("/")
def root():
    return {
        "message": "Face Mask Detection API is running!",
        "usage": "POST image to /predict endpoint"
    }