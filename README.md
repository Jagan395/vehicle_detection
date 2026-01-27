# Vehicle Detection from Aerial Drone Images :
    An end-to-end vehicle detection system designed for aerial drone imagery, capable of identifying vehicles in high-resolution overhead images <br>
The application uses a deep learning–based object detection model served through a FastAPI backend and an interactive frontend UI, fully containerized using Docker.<br>
This project addresses challenges unique to aerial imagery such as small object sizes, scale variation, dense scenes, and occlusions.

---

## Live URL :
   - **TRY IT OUT** <br> 
  
---

## Project Overview:

1.Model training with ultralytics<br>
2.Model serialization using PyTorch <br>
3.RESTful inference API with FastAPI <br>
4.Interactive UI built with Streamlit <br>
5.Dockerized services for reproducibility <br>

---

## How to run:
  - **local** :<br>
    1.install the git <br>
    2.clone the project <br>
    3.install the requirements using the below command <br>
    -**CMD** : pip install -r reqiurements.txt <br>
    4.run the backend code using the below command <br>
    -**CMD** : uvicorn app:app --reload <br>
    5.run the frontend code using the below command <br>
    [Important:Replace the " http://backend:8000/detect " in ui.py with " http://127.0.0.1:8000/detect "] <br>
    -**CMD** : streamlit run u.py <br>

  - **docker**: <br>
      1.install the docker <br>
      2.clone the project using git <br>
      3.install the requirements using the below command <br>
      -**CMD** <br>
       pip install -r reqiurements.txt <br>
      4.build and run the docker image using below command: <br>
     -**CMD**: <br>
        docker compose build up
---
## Author:
   **JAGAN MOHAN REDDY**<br>
   Aspiring Data Scientist
   
   
      
  





