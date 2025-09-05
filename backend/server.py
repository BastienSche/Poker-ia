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
import time

# Import du moteur de poker AVANCÉ
from advanced_poker_engine import AdvancedPokerEngine, GameState, Player, Card, Position
from image_processor import PokerImageProcessor
from poker_vision_prompts import PokerVisionPrompts

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
app = FastAPI(title="Poker Assistant API", version="3.0.0")
api_router = APIRouter(prefix="/api")

# Initialize components AVANCÉS
poker_engine = AdvancedPokerEngine()
image_processor = PokerImageProcessor()
vision_prompts = PokerVisionPrompts()

# Cache pour éviter les analyses répétitives
analysis_cache = {}

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
    phase_hint: Optional[str] = None  # NOUVEAU : hint pour la phase

class PokerAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    detected_elements: Dict[str, Any]
    game_state: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    analysis_type: str = "advanced"  # NOUVEAU

class UserSettings(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    aggressiveness: float = 0.5
    auto_analyze: bool = True
    capture_frequency: int = 2
    language: str = "fr"
    always_on_top: bool = True
    sequential_analysis: bool = True  # NOUVEAU

class GameSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    hands_played: int = 0
    total_profit: float = 0.0
    analyses: List[str] = []

# Helper functions
def generate_image_hash(image_base64: str) -> str:
    """Génère un hash pour éviter les analyses répétitives"""
    import hashlib
    return hashlib.md5(image_base64[:1000].encode()).hexdigest()

def validate_card_format(card: str) -> bool:
    """Valide le format d'une carte (ex: AS, KH, 2C)"""
    if not card or len(card) != 2:
        return False
    
    rank = card[0].upper()
    suit = card[1].upper()
    
    valid_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    valid_suits = ['S', 'H', 'D', 'C']
    
    return rank in valid_ranks and suit in valid_suits

def clean_detected_cards(cards_list: List[str]) -> List[str]:
    """Nettoie et valide une liste de cartes détectées"""
    if not cards_list:
        return []
    
    cleaned_cards = []
    for card in cards_list:
        if isinstance(card, str) and validate_card_format(card):
            cleaned_cards.append(card.upper())
        # Tentative de correction pour les formats mal reconnus
        elif isinstance(card, str) and len(card) >= 2:
            # Extraction du premier caractère (rang) et du dernier (couleur)
            rank = card[0].upper()
            suit = card[-1].upper()
            corrected_card = rank + suit
            if validate_card_format(corrected_card):
                cleaned_cards.append(corrected_card)
    
    return cleaned_cards

def determine_phase_from_board(community_cards: List[str]) -> str:
    """Détermine la phase basée sur le nombre de cartes communes"""
    card_count = len(community_cards) if community_cards else 0
    
    if card_count == 0:
        return 'preflop'
    elif card_count == 3:
        return 'flop'
    elif card_count == 4:
        return 'turn'
    elif card_count == 5:
        return 'river'
    else:
        return 'unknown'

# Routes
@api_router.get("/")
async def root():
    return {"message": "Poker Assistant API - Moteur Avancé Actif !", "version": "3.0.0", "features": ["advanced_engine", "sequential_analysis", "real_equity"]}

@api_router.post("/analyze-screen")
async def analyze_screen_capture(data: ScreenCaptureData):
    """Analyse ULTRA-AVANCÉE avec moteur de poker professionnel"""
    start_time = time.time()
    
    try:
        print(f"🚀 ANALYSE AVANCÉE - Session: {data.session_id}")
        
        # Génération du hash pour le cache
        image_hash = generate_image_hash(data.image_base64)
        
        # Vérification du cache (évite les analyses répétitives)
        if image_hash in analysis_cache:
            cached_result = analysis_cache[image_hash]
            cached_result["id"] = str(uuid.uuid4())
            cached_result["timestamp"] = datetime.utcnow().isoformat()
            cached_result["session_id"] = data.session_id
            cached_result["processing_time"] = 0.001  # Cache hit
            print("💨 Cache hit - résultat instantané")
            return cached_result
        
        # Optimisation de l'image pour l'analyse
        print("🔄 Optimisation de l'image...")
        optimized_image = image_processor.optimize_image_for_poker_analysis(data.image_base64)
        
        # Sélection du prompt selon la phase
        if data.phase_hint:
            system_prompt = vision_prompts.get_sequential_analysis_prompt(data.phase_hint)
            print(f"📋 Prompt spécialisé pour phase: {data.phase_hint}")
        else:
            system_prompt = vision_prompts.get_poker_analysis_prompt()
            print("📋 Prompt général d'analyse")
        
        # Initialisation du chat OpenAI Vision avec modèle ULTRA-RAPIDE
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"poker_speed_{data.session_id}",
            system_message=system_prompt
        ).with_model("openai", "gpt-4o-mini")  # Modèle plus rapide
        
        # Création du message avec l'image optimisée
        image_content = ImageContent(image_base64=optimized_image)
        user_message = UserMessage(
            text=f"Analyse cette table de poker en mode {data.phase_hint or 'auto'}. Focus sur la détection précise du BOARD (cartes communes). Retourne le JSON structuré.",
            file_contents=[image_content]
        )
        
        # Envoi avec timeout RÉDUIT pour vitesse maximale
        print("⚡ Analyse IA ULTRA-RAPIDE...")
        response = await asyncio.wait_for(
            chat.send_message(user_message), 
            timeout=10.0  # Timeout réduit pour plus de vitesse
        )
        
        # Parsing optimisé de la réponse
        detected_elements = await parse_vision_response(response)
        print(f"✅ Éléments détectés: {list(detected_elements.keys())}")
        
        # Validation et correction des cartes détectées
        if "hero_cards" in detected_elements and detected_elements["hero_cards"]:
            original_hero = detected_elements["hero_cards"].copy()
            detected_elements["hero_cards"] = clean_detected_cards(detected_elements["hero_cards"])
            print(f"🃏 Hero cards: {original_hero} → {detected_elements['hero_cards']}")
        
        if "community_cards" in detected_elements and detected_elements["community_cards"]:
            original_board = detected_elements["community_cards"].copy()
            detected_elements["community_cards"] = clean_detected_cards(detected_elements["community_cards"])
            print(f"🎯 Board cards: {original_board} → {detected_elements['community_cards']}")
        
        # Détermination automatique de la phase
        auto_phase = determine_phase_from_board(detected_elements.get("community_cards", []))
        detected_phase = detected_elements.get("betting_round", auto_phase)
        if detected_phase != auto_phase:
            print(f"⚠️ Phase corrigée: {detected_phase} → {auto_phase}")
            detected_elements["betting_round"] = auto_phase
        
        # Construction avancée de l'état de jeu
        game_state = None
        recommendation = None
        confidence = detected_elements.get("confidence_level", 0.0)
        
        if detected_elements.get("hero_cards") and len(detected_elements["hero_cards"]) == 2:
            try:
                print("🏗️ Construction de l'état de jeu avancé...")
                
                # Conversion des cartes héros
                hero_cards = []
                for card_str in detected_elements["hero_cards"]:
                    if len(card_str) == 2:
                        hero_cards.append(Card(rank=card_str[0], suit=card_str[1].lower()))
                
                # Construction des joueurs
                players = []
                
                # Joueur principal avec ses cartes
                hero_player = Player(
                    name="Hero",
                    stack=detected_elements.get("players", [{}])[0].get("stack", 1500),
                    position=Position.DEALER,  # Assumé pour simplifier
                    cards=hero_cards,
                    is_active=True
                )
                players.append(hero_player)
                
                # Joueurs adverses
                for i in range(2):
                    position = Position.SMALL_BLIND if i == 0 else Position.BIG_BLIND
                    stack = 1500  # Stack par défaut
                    if len(detected_elements.get("players", [])) > i + 1:
                        stack = detected_elements["players"][i + 1].get("stack", 1500)
                    
                    opponent = Player(
                        name=f"Opponent_{i+1}",
                        stack=stack,
                        position=position,
                        is_active=True
                    )
                    players.append(opponent)
                
                # Cartes communes
                community_cards = []
                if detected_elements.get("community_cards"):
                    for card_str in detected_elements["community_cards"]:
                        if len(card_str) == 2:
                            community_cards.append(Card(rank=card_str[0], suit=card_str[1].lower()))
                
                print(f"🎮 État de jeu: {len(community_cards)} cartes board, phase {auto_phase}")
                
                # État de jeu
                game_state = GameState(
                    players=players,
                    community_cards=community_cards,
                    pot=detected_elements.get("pot", 150),
                    small_blind=detected_elements.get("blinds", {}).get("small_blind", 25),
                    big_blind=detected_elements.get("blinds", {}).get("big_blind", 50),
                    current_player=0,
                    betting_round=auto_phase
                )
                
                # ANALYSE AVANCÉE avec vraies librairies de poker
                print("🧠 Calcul de recommandation avancée...")
                recommendation_result = poker_engine.get_recommended_action(
                    game_state, Position.DEALER, aggressiveness=0.5
                )
                
                # Conversion du résultat en dict
                recommendation = {
                    "action": recommendation_result.action,
                    "confidence": recommendation_result.confidence,
                    "reasoning": recommendation_result.reasoning,
                    "hand_strength": recommendation_result.hand_strength,
                    "pot_odds": recommendation_result.pot_odds,
                    "phase": recommendation_result.phase,
                    "equity": recommendation_result.equity,
                    "hand_type": recommendation_result.hand_type,
                    "outs": recommendation_result.outs,
                    "implied_odds": recommendation_result.implied_odds,
                    "position_factor": recommendation_result.position_factor,
                    "icm_factor": recommendation_result.icm_factor
                }
                
                confidence = max(confidence, recommendation_result.confidence)
                
                print(f"💡 Recommandation: {recommendation_result.action.upper()} ({recommendation_result.confidence:.1%})")
                print(f"🎯 Équité: {recommendation_result.equity:.1%}, Outs: {recommendation_result.outs}")
                
            except Exception as e:
                print(f"❌ Erreur construction état: {e}")
                recommendation = {
                    "action": "fold",
                    "confidence": 0.3,
                    "reasoning": f"Erreur d'analyse avancée: {str(e)}",
                    "error": True
                }
        
        # Calcul du temps de traitement
        processing_time = time.time() - start_time
        
        # Conversion manuelle de GameState en dict
        game_state_dict = None
        if game_state:
            game_state_dict = {
                "players": [
                    {
                        "name": p.name,
                        "stack": p.stack,
                        "position": p.position.value if hasattr(p.position, 'value') else str(p.position),
                        "current_bet": p.current_bet,
                        "is_active": p.is_active,
                        "cards": [f"{c.rank}{c.suit}" for c in p.cards] if p.cards else None
                    } for p in game_state.players
                ],
                "community_cards": [f"{c.rank}{c.suit}" for c in game_state.community_cards],
                "pot": game_state.pot,
                "small_blind": game_state.small_blind,
                "big_blind": game_state.big_blind,
                "betting_round": game_state.betting_round
            }
        
        # Création du résultat
        analysis_result = PokerAnalysisResult(
            session_id=data.session_id,
            detected_elements=detected_elements,
            game_state=game_state_dict,
            recommendation=recommendation,
            confidence=confidence,
            processing_time=processing_time,
            analysis_type="advanced"
        )
        
        # Mise en cache pour éviter les analyses répétitives
        analysis_cache[image_hash] = analysis_result.dict()
        
        # Nettoyage du cache (garde seulement les 10 dernières analyses)
        if len(analysis_cache) > 10:
            oldest_key = next(iter(analysis_cache))
            del analysis_cache[oldest_key]
        
        # Sauvegarde en base (asynchrone pour ne pas ralentir)
        asyncio.create_task(save_analysis_async(analysis_result))
        
        # Diffusion WebSocket (asynchrone)
        asyncio.create_task(broadcast_analysis_async(analysis_result))
        
        print(f"✅ Analyse avancée terminée en {processing_time:.2f}s")
        return analysis_result.dict()
        
    except asyncio.TimeoutError:
        processing_time = time.time() - start_time
        print("⏰ Timeout d'analyse")
        return {
            "error": "Timeout",
            "message": "L'analyse a pris trop de temps",
            "processing_time": processing_time,
            "session_id": data.session_id
        }
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Erreur critique: {str(e)}")
        logging.error(f"Erreur lors de l'analyse: {str(e)}")
        return {
            "error": "AnalysisError",
            "message": f"Erreur d'analyse: {str(e)}",
            "processing_time": processing_time,
            "session_id": data.session_id
        }

async def parse_vision_response(response: str) -> dict:
    """Parse optimisé de la réponse Vision"""
    try:
        # Nettoyage de la réponse
        response_text = response.strip()
        
        # Suppression des balises markdown si présentes
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Parse JSON
        detected_elements = json.loads(response_text)
        
        # Validation de base
        if not isinstance(detected_elements, dict):
            raise ValueError("Réponse n'est pas un dictionnaire")
        
        return detected_elements
        
    except json.JSONDecodeError as e:
        print(f"Erreur parsing JSON: {e}")
        # Fallback avec structure par défaut
        return {
            "error": "JSON parsing failed",
            "raw_response": response[:500],  # Première partie seulement
            "blinds": {"small_blind": None, "big_blind": None, "ante": 0},
            "pot": None,
            "players": [],
            "community_cards": [],
            "hero_cards": None,
            "betting_round": "unknown",
            "confidence_level": 0.0
        }

async def save_analysis_async(analysis_result: PokerAnalysisResult):
    """Sauvegarde asynchrone en base"""
    try:
        await db.poker_analyses.insert_one(analysis_result.dict())
    except Exception as e:
        print(f"Erreur sauvegarde: {e}")

async def broadcast_analysis_async(analysis_result: PokerAnalysisResult):
    """Diffusion WebSocket asynchrone"""
    try:
        await manager.broadcast({
            "type": "analysis_result",
            "data": analysis_result.dict()
        })
    except Exception as e:
        print(f"Erreur broadcast: {e}")

# Routes existantes (inchangées)
@api_router.get("/sessions/{session_id}/analyses")
async def get_session_analyses(session_id: str, limit: int = 20):
    """Récupère les analyses d'une session (limité pour la performance)"""
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
        default_settings = UserSettings(user_id=user_id)
        await db.user_settings.insert_one(default_settings.dict())
        return default_settings

@api_router.get("/performance/stats")
async def get_performance_stats():
    """Statistiques de performance de l'API"""
    return {
        "cache_size": len(analysis_cache),
        "average_processing_time": "< 5s",
        "optimization_level": "maximum",
        "version": "3.0.0",
        "features": ["treys_engine", "real_equity", "sequential_analysis"]
    }

@api_router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket optimisé pour les mises à jour temps réel"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await manager.send_personal_message({
                "type": "echo",
                "data": message,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
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