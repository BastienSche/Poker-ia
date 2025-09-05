from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import base64
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
from io import BytesIO
from PIL import Image

# Import du moteur de poker
from poker_engine import PokerEngine, GameState, Player, Card, Position

# Import de l'intégration LLM
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize OpenAI Vision
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Create the main app
app = FastAPI(title="Poker Assistant API", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Initialize poker engine
poker_engine = PokerEngine()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

# Models
class ScreenCaptureData(BaseModel):
    image_base64: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str

class PokerAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    detected_elements: Dict[str, Any]
    game_state: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

class UserSettings(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    aggressiveness: float = 0.5  # 0.0 = très conservateur, 1.0 = très agressif
    auto_analyze: bool = True
    capture_frequency: int = 2  # secondes
    language: str = "fr"
    always_on_top: bool = True

class GameSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    hands_played: int = 0
    total_profit: float = 0.0
    analyses: List[str] = []  # IDs des analyses

# Routes
@api_router.get("/")
async def root():
    return {"message": "Poker Assistant API - Prêt pour l'analyse !"}

@api_router.post("/analyze-screen")
async def analyze_screen_capture(data: ScreenCaptureData):
    """Analyse une capture d'écran et retourne les éléments détectés + recommandation"""
    try:
        # Décodage de l'image base64
        image_data = base64.b64decode(data.image_base64)
        
        # Initialisation du chat OpenAI Vision
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"poker_analysis_{data.session_id}",
            system_message="""Tu es un expert en analyse de tables de poker Texas Hold'em Spin & Go 3 joueurs.
            
Analyse cette image de table de poker et retourne UNIQUEMENT un JSON structuré avec les informations suivantes :

{
  "blinds": {
    "small_blind": <valeur>,
    "big_blind": <valeur>,
    "ante": <valeur ou 0>
  },
  "pot": <taille du pot>,
  "players": [
    {
      "position": "dealer|small_blind|big_blind",
      "name": "<pseudo>",
      "stack": <taille du stack>,
      "current_bet": <mise actuelle>,
      "last_action": "fold|call|raise|check|null",
      "is_active": true/false
    }
  ],
  "community_cards": ["<carte1>", "<carte2>", "<carte3>", "<carte4>", "<carte5>"],
  "hero_cards": ["<carte1>", "<carte2>"],
  "betting_round": "preflop|flop|turn|river"
}

Format des cartes : utilise "AS" pour As de Pique, "KH" pour Roi de Cœur, "QD" pour Dame de Carreau, "JC" pour Valet de Trèfle, etc.
S=Spades/Pique, H=Hearts/Cœur, D=Diamonds/Carreau, C=Clubs/Trèfle

Si tu ne peux pas détecter certains éléments, utilise null ou des valeurs par défaut appropriées."""
        ).with_model("openai", "gpt-4o")
        
        # Création du message avec l'image
        image_content = ImageContent(image_base64=data.image_base64)
        user_message = UserMessage(
            text="Analyse cette table de poker et retourne les informations au format JSON demandé.",
            file_contents=[image_content]
        )
        
        # Envoi de la requête
        response = await chat.send_message(user_message)
        
        # Parsing de la réponse JSON
        try:
            # Extraction du JSON de la réponse
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            detected_elements = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback si le parsing JSON échoue
            detected_elements = {
                "error": "Impossible de parser la réponse JSON",
                "raw_response": response,
                "blinds": {"small_blind": 0, "big_blind": 0, "ante": 0},
                "pot": 0,
                "players": [],
                "community_cards": [],
                "hero_cards": [],
                "betting_round": "preflop"
            }
        
        # Construction de l'état de jeu pour l'analyse stratégique
        game_state = None
        recommendation = None
        confidence = 0.0
        
        if "hero_cards" in detected_elements and detected_elements["hero_cards"]:
            try:
                # Conversion des données détectées en objets du moteur de poker
                players = []
                for player_data in detected_elements.get("players", []):
                    position_map = {
                        "dealer": Position.DEALER,
                        "small_blind": Position.SMALL_BLIND,
                        "big_blind": Position.BIG_BLIND
                    }
                    
                    position = position_map.get(player_data.get("position"), Position.DEALER)
                    
                    player = Player(
                        name=player_data.get("name", "Unknown"),
                        stack=player_data.get("stack", 0),
                        position=position,
                        current_bet=player_data.get("current_bet", 0),
                        is_active=player_data.get("is_active", True)
                    )
                    
                    # Ajout des cartes pour le héros
                    if player_data.get("position") in ["dealer", "small_blind", "big_blind"]:
                        hero_cards_data = detected_elements.get("hero_cards", [])
                        if hero_cards_data and len(hero_cards_data) == 2:
                            player.cards = poker_engine.parse_cards_from_text(" ".join(hero_cards_data))
                    
                    players.append(player)
                
                # Construction de l'état de jeu
                community_cards = poker_engine.parse_cards_from_text(" ".join(detected_elements.get("community_cards", [])))
                
                game_state = GameState(
                    players=players,
                    community_cards=community_cards,
                    pot=detected_elements.get("pot", 0),
                    small_blind=detected_elements.get("blinds", {}).get("small_blind", 0),
                    big_blind=detected_elements.get("blinds", {}).get("big_blind", 0),
                    current_player=0,
                    betting_round=detected_elements.get("betting_round", "preflop")
                )
                
                # Obtenir la recommandation stratégique
                # Pour simplifier, on assume que le héros est en position dealer
                recommendation = poker_engine.get_recommended_action(
                    game_state, Position.DEALER, aggressiveness=0.5
                )
                confidence = recommendation.get("confidence", 0.0)
                
            except Exception as e:
                recommendation = {
                    "action": "fold",
                    "confidence": 0.0,
                    "reasoning": f"Erreur dans l'analyse stratégique: {str(e)}",
                    "error": True
                }
        
        # Sauvegarde de l'analyse
        analysis_result = PokerAnalysisResult(
            session_id=data.session_id,
            detected_elements=detected_elements,
            game_state=game_state.dict() if game_state else None,
            recommendation=recommendation,
            confidence=confidence
        )
        
        # Insertion en base
        await db.poker_analyses.insert_one(analysis_result.dict())
        
        # Diffusion via WebSocket
        await manager.broadcast({
            "type": "analysis_result",
            "data": analysis_result.dict()
        })
        
        return analysis_result.dict()
        
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {str(e)}")

@api_router.get("/sessions/{session_id}/analyses")
async def get_session_analyses(session_id: str, limit: int = 50):
    """Récupère les analyses d'une session"""
    analyses = await db.poker_analyses.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    
    return [PokerAnalysisResult(**analysis) for analysis in analyses]

@api_router.post("/settings")
async def save_user_settings(settings: UserSettings):
    """Sauvegarde les paramètres utilisateur"""
    await db.user_settings.replace_one(
        {"user_id": settings.user_id},
        settings.dict(),
        upsert=True
    )
    return {"status": "saved"}

@api_router.get("/settings/{user_id}")
async def get_user_settings(user_id: str):
    """Récupère les paramètres utilisateur"""
    settings = await db.user_settings.find_one({"user_id": user_id})
    if settings:
        return UserSettings(**settings)
    else:
        # Paramètres par défaut
        default_settings = UserSettings(user_id=user_id)
        await db.user_settings.insert_one(default_settings.dict())
        return default_settings

@api_router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket pour les mises à jour temps réel"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Echo du message pour confirmer la réception
            await manager.send_personal_message({
                "type": "echo",
                "data": message,
                "session_id": session_id
            }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Include router
app.include_router(api_router)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)