from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_TITLE: str = 'Helper App'
    APP_VERSION: str = '0.1.0'
    DEBUG: bool = True
    HOST: str = '127.0.0.1'
    PORT: int = 8000

    class Config:
        env_file = '.env'

settings = Settings()
