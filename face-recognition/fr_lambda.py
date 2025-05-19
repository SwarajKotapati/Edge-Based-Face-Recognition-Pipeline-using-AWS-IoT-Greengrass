import os
import json
import torch
import base64
import boto3
import numpy as np
from PIL import Image
from io import BytesIO

# AWS SQS Config
SQS_CLIENT = boto3.client("sqs")
REQUEST_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/861276114572/1230469840-req-queue"
RESPONSE_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/861276114572/1230469840-resp-queue"

MODEL_PATH = '/mnt/efs/resnetV1.pt'
WEIGHTS_PATH = '/mnt/efs/resnetV1_video_weights.pt'

# Load ResNet Model
print("🔄 Loading ResNet Model...")
resnet = torch.jit.load(MODEL_PATH)
saved_data = torch.load(WEIGHTS_PATH)
embedding_list = saved_data[0]  # Face embeddings
name_list = saved_data[1]  # Names

def recognize_face(face_img_base64):
    # print("Recognize_face Function entry ", face_img_base64)
    """Recognizes a face from Base64-encoded image"""
    
    # Decode Base64 to Image
    face_img_data = base64.b64decode(face_img_base64, validate=True)
    face_pil = Image.open(BytesIO(face_img_data)).convert("RGB")

    # Convert Image to NumPy Array
    face_numpy = np.array(face_pil, dtype=np.float32) / 255.0  # Normalize [0,1]
    face_numpy = np.transpose(face_numpy, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)
    
    # Convert to PyTorch Tensor
    face_tensor = torch.tensor(face_numpy, dtype=torch.float32).unsqueeze(0)  # Add batch dim

    # Compute Face Embedding
    emb = resnet(face_tensor).detach()  # Disable gradient computation

    # Find Closest Match
    dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
    idx_min = dist_list.index(min(dist_list))

    return name_list[idx_min]  # Return the closest matching name

def lambda_handler(event, context):
    """AWS Lambda Entry Point"""

    print(f"📥 Processing {len(event['Records'])} messages...")

    results = []
    
    for record in event['Records']:
        body = json.loads(record['body'])
        request_id = body.get("request_id")
        face_img_base64 = body.get("face_image")

        if not request_id or not face_img_base64:
            print("❌ Skipping message: Missing data")
            continue

        print(f"🔍 Processing request: {request_id}")

        # Recognize Face
        predicted_name = recognize_face(face_img_base64)

        print(f"✅ Recognized as: {predicted_name}")

        # Store result
        results.append({"request_id": request_id, "result": predicted_name})

    # Batch Send Results to SQS
    if results:
        entries = [
            {"Id": str(i), "MessageBody": json.dumps(result)}
            for i, result in enumerate(results)
        ]
        SQS_CLIENT.send_message_batch(QueueUrl=RESPONSE_QUEUE_URL, Entries=entries)
        print(f"📤 Sent {len(results)} results to response queue.")

    return {"statusCode": 200, "body": json.dumps({"message": "Batch processed requests"})}

