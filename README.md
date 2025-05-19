# Edge-Based Face Recognition Pipeline using AWS IoT Greengrass

This project is a distributed, low-latency face recognition system built using AWS edge and cloud services. It was developed as part of **CSE546 - Cloud Computing** at Arizona State University.

## Overview

The system processes video frames from IoT-like devices by running face detection at the edge and face recognition in the cloud. AWS IoT Greengrass, Lambda, and SQS are used to implement a scalable and privacy-aware pipeline for real-time ML inference.

## Architecture

![image](https://github.com/user-attachments/assets/88845188-5509-414f-b3ed-12c9ba2939a1)

The application consists of an IoT client device (EC2) publishing video frames to an MQTT topic. A Greengrass Core device (EC2) performs local face detection (MTCNN). Detected faces are sent to an SQS request queue, triggering a Lambda function that performs face recognition (FaceNet). Results are returned via an SQS response queue.

![image](https://github.com/user-attachments/assets/bb7ea27d-23b2-4177-8617-fe008f388118)


## Features

- Built a full edge-cloud ML pipeline using AWS IoT Greengrass v2, EC2, Lambda, and SQS
- Performed real-time face detection at the edge with MTCNN
- Offloaded face recognition to cloud Lambda using FaceNet
- Used MQTT and SQS for communication between edge and cloud components
- Skipped unnecessary Lambda calls on no-face frames to reduce latency and cost
- Tested and validated using automated grading infrastructure

## Technologies Used

- AWS IoT Greengrass v2  
- Amazon EC2, Lambda, SQS  
- MQTT via AWS IoT Core  
- Python 3, PyTorch, MTCNN, FaceNet  
- AWS CLI, Boto3

## Key Metrics

| Metric                          | Value        |
|----------------------------------|--------------|
| Requests Processed               | 100          |
| Success Rate                     | 100%         |
| Classification Accuracy          | 100%         |
| Average Latency per Request      | < 1 second   |
| Cloud Calls Skipped (No-Face)    | ~50%         |
| Failed Messages                  | 0            |

## How It Works

1. IoT client device publishes video frames to an MQTT topic.  
2. Greengrass Core subscribes to the topic and runs MTCNN face detection.  
3. If a face is detected, the image is sent to an SQS request queue.  
4. AWS Lambda is triggered and performs FaceNet-based recognition.  
5. The recognition result is sent to an SQS response queue and consumed by the client.  
6. If no face is detected, the edge device directly writes a `"No-Face"` result to the response queue (bonus optimization).

## Project Structure

![image](https://github.com/user-attachments/assets/3ff865d1-0609-434b-acc4-7eb969fe12e2)


## Acknowledgment

Developed by [Venkata Swaraj Kotapati] @ Arizona State University.
