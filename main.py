from fastapi import FastAPI
from config.database import engine
from models.models import Base
from routes.users import router as user_router

app = FastAPI()

# Create database tables
Base.metadata.create_all(bind=engine)

# Include routes
app.include_router(user_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Service Provider Platform API"}
