import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import { 
  Play, 
  Square, 
  Settings, 
  Monitor, 
  Camera, 
  TrendingUp, 
  AlertCircle,
  Zap,
  Target,
  Brain,
  Eye,
  Clock,
  Activity
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;
const WS_URL = `${BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/api/ws`;

function App() {
  // États principaux
  const [isCapturing, setIsCapturing] = useState(false);
  const [stream, setStream] = useState(null);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [settings, setSettings] = useState({
    aggressiveness: 0.5,
    autoAnalyze: true,
    captureFrequency: 2,
    alwaysOnTop: true
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [stats, setStats] = useState({
    handsAnalyzed: 0,
    avgConfidence: 0,
    lastUpdateTime: null
  });
  
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  // Connexion WebSocket
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        wsRef.current = new WebSocket(`${WS_URL}/${sessionId}`);
        
        wsRef.current.onopen = () => {
          setConnectionStatus('connected');
          console.log('WebSocket connecté');
        };
        
        wsRef.current.onmessage = (event) => {
          const message = JSON.parse(event.data);
          if (message.type === 'analysis_result') {
            setCurrentAnalysis(message.data);
            setAnalysisHistory(prev => [message.data, ...prev.slice(0, 9)]);
            setStats(prev => ({
              handsAnalyzed: prev.handsAnalyzed + 1,
              avgConfidence: (prev.avgConfidence + message.data.confidence) / 2,
              lastUpdateTime: new Date().toLocaleTimeString()
            }));
          }
        };
        
        wsRef.current.onclose = () => {
          setConnectionStatus('disconnected');
          setTimeout(connectWebSocket, 3000);
        };
        
        wsRef.current.onerror = () => {
          setConnectionStatus('error');
        };
      } catch (error) {
        console.error('Erreur WebSocket:', error);
        setConnectionStatus('error');
      }
    };

    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [sessionId]);

  // Fonction de capture d'écran
  const captureScreen = useCallback(async () => {
    if (!stream || !videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    // Conversion en base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    const base64Data = imageData.split(',')[1];
    
    try {
      const response = await fetch(`${API}/analyze-screen`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: base64Data,
          session_id: sessionId
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        setCurrentAnalysis(result);
        setAnalysisHistory(prev => [result, ...prev.slice(0, 9)]);
      }
    } catch (error) {
      console.error('Erreur analyse:', error);
    }
  }, [stream, sessionId]);

  // Démarrage de la capture
  const startCapture = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          mediaSource: 'screen',
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });
      
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      
      setIsCapturing(true);
      
      // Capture automatique selon la fréquence
      if (settings.autoAnalyze) {
        intervalRef.current = setInterval(captureScreen, settings.captureFrequency * 1000);
      }
      
    } catch (error) {
      console.error('Erreur capture:', error);
      alert('Impossible d\'accéder à la capture d\'écran. Vérifiez les autorisations.');
    }
  };

  // Arrêt de la capture
  const stopCapture = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setIsCapturing(false);
  };

  // Nettoyage
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  // Rendu de la recommandation
  const renderRecommendation = (recommendation) => {
    if (!recommendation) return null;

    const actionColors = {
      fold: 'text-red-400 bg-red-900/20 border-red-500/30',
      call: 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30',
      raise: 'text-green-400 bg-green-900/20 border-green-500/30',
      all_in: 'text-purple-400 bg-purple-900/20 border-purple-500/30'
    };

    const actionIcons = {
      fold: <AlertCircle className="w-5 h-5" />,
      call: <Clock className="w-5 h-5" />,
      raise: <TrendingUp className="w-5 h-5" />,
      all_in: <Zap className="w-5 h-5" />
    };

    return (
      <div className={`p-4 rounded-lg border-2 ${actionColors[recommendation.action] || 'text-gray-400 bg-gray-900/20 border-gray-500/30'}`}>
        <div className="flex items-center gap-3 mb-2">
          {actionIcons[recommendation.action]}
          <h3 className="text-xl font-bold uppercase">
            {recommendation.action === 'fold' ? 'COUCHER' :
             recommendation.action === 'call' ? 'SUIVRE' :
             recommendation.action === 'raise' ? 'RELANCER' :
             recommendation.action === 'all_in' ? 'TAPIS' : recommendation.action}
          </h3>
          <div className="flex-1" />
          <div className={`px-2 py-1 rounded text-sm font-medium ${
            recommendation.confidence > 0.7 ? 'bg-green-900/30 text-green-300' :
            recommendation.confidence > 0.4 ? 'bg-yellow-900/30 text-yellow-300' :
            'bg-red-900/30 text-red-300'
          }`}>
            {Math.round(recommendation.confidence * 100)}% confiance
          </div>
        </div>
        <p className="text-sm opacity-90">{recommendation.reasoning}</p>
        {recommendation.hand_strength && (
          <div className="flex gap-4 mt-2 text-xs">
            <span>Force: {Math.round(recommendation.hand_strength * 100)}%</span>
            {recommendation.pot_odds > 0 && (
              <span>Pot Odds: {recommendation.pot_odds.toFixed(1)}:1</span>
            )}
            <span>Phase: {recommendation.phase}</span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur border-b border-slate-700">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold">Assistant Poker Pro</h1>
                <p className="text-sm text-slate-400">Analyse Texas Hold'em Spin & Go</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
                connectionStatus === 'connected' ? 'bg-green-900/30 text-green-300' :
                connectionStatus === 'error' ? 'bg-red-900/30 text-red-300' :
                'bg-yellow-900/30 text-yellow-300'
              }`}>
                <Activity className="w-4 h-4" />
                <span className="text-sm">
                  {connectionStatus === 'connected' ? 'Connecté' :
                   connectionStatus === 'error' ? 'Erreur' : 'Déconnecté'}
                </span>
              </div>
              
              <button
                onClick={() => setIsSettingsOpen(true)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Panel de capture principal */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Contrôles de capture */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Eye className="w-5 h-5 text-blue-400" />
                  <h2 className="text-lg font-semibold">Capture d'Écran</h2>
                </div>
                <div className="flex gap-2">
                  {!isCapturing ? (
                    <button
                      onClick={startCapture}
                      className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors font-medium"
                    >
                      <Play className="w-4 h-4" />
                      Démarrer
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={captureScreen}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors font-medium"
                      >
                        <Camera className="w-4 h-4" />
                        Analyser
                      </button>
                      <button
                        onClick={stopCapture}
                        className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors font-medium"
                      >
                        <Square className="w-4 h-4" />
                        Arrêter
                      </button>
                    </>
                  )}
                </div>
              </div>
              
              {/* Aperçu vidéo */}
              <div className="relative bg-slate-900 rounded-lg overflow-hidden" style={{aspectRatio: '16/9'}}>
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  className="w-full h-full object-contain"
                  style={{display: isCapturing ? 'block' : 'none'}}
                />
                {!isCapturing && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <Monitor className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                      <p className="text-slate-400">Cliquez sur "Démarrer" pour capturer votre écran</p>
                    </div>
                  </div>
                )}
                <canvas ref={canvasRef} className="hidden" />
              </div>
            </div>

            {/* Résultats de l'analyse */}
            {currentAnalysis && (
              <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
                <div className="flex items-center gap-3 mb-4">
                  <Target className="w-5 h-5 text-green-400" />
                  <h2 className="text-lg font-semibold">Analyse en Cours</h2>
                  <div className="text-sm text-slate-400">
                    {new Date(currentAnalysis.timestamp).toLocaleTimeString()}
                  </div>
                </div>
                
                {currentAnalysis.recommendation && renderRecommendation(currentAnalysis.recommendation)}
                
                {/* Détails détectés */}
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                  {currentAnalysis.detected_elements?.pot > 0 && (
                    <div className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="text-sm text-slate-400 mb-1">Pot</div>
                      <div className="font-semibold">{currentAnalysis.detected_elements.pot}</div>
                    </div>
                  )}
                  
                  {currentAnalysis.detected_elements?.blinds && (
                    <div className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="text-sm text-slate-400 mb-1">Blinds</div>
                      <div className="font-semibold">
                        {currentAnalysis.detected_elements.blinds.small_blind}/
                        {currentAnalysis.detected_elements.blinds.big_blind}
                      </div>
                    </div>
                  )}
                  
                  {currentAnalysis.detected_elements?.hero_cards?.length > 0 && (
                    <div className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="text-sm text-slate-400 mb-1">Vos Cartes</div>
                      <div className="font-semibold font-mono">
                        {currentAnalysis.detected_elements.hero_cards.join(' ')}
                      </div>
                    </div>
                  )}
                  
                  {currentAnalysis.detected_elements?.community_cards?.length > 0 && (
                    <div className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="text-sm text-slate-400 mb-1">Board</div>
                      <div className="font-semibold font-mono">
                        {currentAnalysis.detected_elements.community_cards.join(' ')}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Panel latéral */}
          <div className="space-y-6">
            
            {/* Statistiques */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-5 h-5 text-purple-400" />
                <h2 className="text-lg font-semibold">Statistiques</h2>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-400">Mains analysées</span>
                  <span className="font-semibold">{stats.handsAnalyzed}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Confiance moy.</span>
                  <span className="font-semibold">{Math.round(stats.avgConfidence * 100)}%</span>
                </div>
                {stats.lastUpdateTime && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Dernière analyse</span>
                    <span className="font-semibold text-sm">{stats.lastUpdateTime}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Historique des analyses */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-4">
                <Clock className="w-5 h-5 text-orange-400" />
                <h2 className="text-lg font-semibold">Historique</h2>
              </div>
              
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {analysisHistory.length === 0 ? (
                  <p className="text-slate-400 text-sm">Aucune analyse encore</p>
                ) : (
                  analysisHistory.map((analysis, index) => (
                    <div key={analysis.id || index} className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium">
                          {analysis.recommendation?.action?.toUpperCase() || 'N/A'}
                        </span>
                        <span className="text-xs text-slate-400">
                          {new Date(analysis.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="text-xs text-slate-400">
                        Confiance: {Math.round((analysis.confidence || 0) * 100)}%
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modal des paramètres */}
      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl p-6 max-w-md w-full mx-4 border border-slate-700">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold">Paramètres</h2>
              <button
                onClick={() => setIsSettingsOpen(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Agressivité: {Math.round(settings.aggressiveness * 100)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.aggressiveness}
                  onChange={(e) => setSettings({...settings, aggressiveness: parseFloat(e.target.value)})}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                  <span>Conservateur</span>
                  <span>Agressif</span>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  Fréquence d'analyse (secondes)
                </label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={settings.captureFrequency}
                  onChange={(e) => setSettings({...settings, captureFrequency: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 bg-slate-700 rounded-lg border border-slate-600"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Analyse automatique</span>
                <input
                  type="checkbox"
                  checked={settings.autoAnalyze}
                  onChange={(e) => setSettings({...settings, autoAnalyze: e.target.checked})}
                  className="w-4 h-4"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Toujours au premier plan</span>
                <input
                  type="checkbox"
                  checked={settings.alwaysOnTop}
                  onChange={(e) => setSettings({...settings, alwaysOnTop: e.target.checked})}
                  className="w-4 h-4"
                />
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  // Sauvegarde des paramètres
                  setIsSettingsOpen(false);
                }}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors font-medium"
              >
                Sauvegarder
              </button>
              <button
                onClick={() => setIsSettingsOpen(false)}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              >
                Annuler
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;