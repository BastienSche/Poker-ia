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
  Activity,
  CheckCircle,
  XCircle,
  Loader,
  Cpu,
  Gauge
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;
const WS_URL = `${BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/api/ws`;

function App() {
  // États principaux
  const [isCapturing, setIsCapturing] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [stream, setStream] = useState(null);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [settings, setSettings] = useState({
    aggressiveness: 0.5,
    autoAnalyze: true,
    captureFrequency: 2,
    alwaysOnTop: true,
    useLocalAI: false, // Désactivé par défaut pour éviter les erreurs
    continuousAnalysis: false
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [stats, setStats] = useState({
    handsAnalyzed: 0,
    avgConfidence: 0,
    lastUpdateTime: null,
    avgProcessingTime: 0
  });
  const [lastAnalysisTime, setLastAnalysisTime] = useState(null);
  
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
          console.log('✅ WebSocket connecté');
        };
        
        wsRef.current.onmessage = (event) => {
          const message = JSON.parse(event.data);
          if (message.type === 'analysis_result') {
            handleAnalysisResult(message.data);
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

  // Gestionnaire de résultats d'analyse - CORRECTION DU BUG
  const handleAnalysisResult = useCallback((analysisData) => {
    setCurrentAnalysis(analysisData);
    setAnalysisHistory(prev => [analysisData, ...prev.slice(0, 19)]);
    
    // Mise à jour correcte des statistiques
    setStats(prev => {
      const newCount = prev.handsAnalyzed + 1;
      const confidence = analysisData.confidence || 0;
      const processingTime = analysisData.processing_time || 0;
      
      const newAvgConfidence = prev.handsAnalyzed === 0 
        ? confidence
        : (prev.avgConfidence * prev.handsAnalyzed + confidence) / newCount;
      
      const newAvgTime = prev.handsAnalyzed === 0
        ? processingTime
        : (prev.avgProcessingTime * prev.handsAnalyzed + processingTime) / newCount;

      return {
        handsAnalyzed: newCount,
        avgConfidence: newAvgConfidence,
        lastUpdateTime: new Date().toLocaleTimeString(),
        avgProcessingTime: newAvgTime
      };
    });
    
    setIsAnalyzing(false);
    setLastAnalysisTime(new Date());
  }, []);

  // Fonction d'analyse cloud simplifiée
  const analyzeScreen = useCallback(async () => {
    if (!stream || !videoRef.current || !canvasRef.current || isAnalyzing) {
      return;
    }

    setIsAnalyzing(true);
    setLastAnalysisTime(new Date());

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    // Optimisation : résolution réduite
    const targetWidth = 1280;
    const targetHeight = 720;
    
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
    
    // Conversion optimisée en base64
    const imageData = canvas.toDataURL('image/jpeg', 0.75);
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
        if (!result.error) {
          handleAnalysisResult(result);
        } else {
          console.error('Erreur API:', result.message);
          setIsAnalyzing(false);
        }
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Erreur analyse:', error);
      setIsAnalyzing(false);
      
      // Affichage d'erreur temporaire
      setCurrentAnalysis({
        error: true,
        message: `Erreur: ${error.message}`,
        timestamp: new Date().toISOString()
      });
      
      setTimeout(() => {
        setCurrentAnalysis(null);
      }, 5000);
    }
  }, [stream, sessionId, isAnalyzing, handleAnalysisResult]);

  // Démarrage de la capture
  const startCapture = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          mediaSource: 'screen',
          width: { ideal: 1920, max: 1920 },
          height: { ideal: 1080, max: 1080 },
          frameRate: { ideal: 30, max: 30 }
        },
        audio: false
      });
      
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      
      setIsCapturing(true);
      
      // Capture automatique selon paramètres
      if (settings.autoAnalyze) {
        const frequency = settings.continuousAnalysis ? 1000 : settings.captureFrequency * 1000;
        intervalRef.current = setInterval(() => {
          if (!isAnalyzing) {
            analyzeScreen();
          }
        }, frequency);
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
    setIsAnalyzing(false);
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

  // Rendu optimisé de la recommandation
  const renderRecommendation = (recommendation) => {
    if (!recommendation) return null;

    const actionColors = {
      fold: 'text-red-400 bg-red-900/20 border-red-500/30',
      call: 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30',
      raise: 'text-green-400 bg-green-900/20 border-green-500/30',
      all_in: 'text-purple-400 bg-purple-900/20 border-purple-500/30'
    };

    const actionIcons = {
      fold: <XCircle className="w-6 h-6" />,
      call: <Clock className="w-6 h-6" />,
      raise: <TrendingUp className="w-6 h-6" />,
      all_in: <Zap className="w-6 h-6" />
    };

    const actionLabels = {
      fold: 'COUCHER',
      call: 'SUIVRE', 
      raise: 'RELANCER',
      all_in: 'TAPIS'
    };

    return (
      <div className={`p-6 rounded-xl border-2 ${actionColors[recommendation.action] || 'text-gray-400 bg-gray-900/20 border-gray-500/30'} recommendation-card`}>
        <div className="flex items-center gap-4 mb-3">
          {actionIcons[recommendation.action]}
          <h3 className="text-2xl font-bold">
            {actionLabels[recommendation.action] || recommendation.action?.toUpperCase()}
          </h3>
          <div className="flex-1" />
          <div className={`px-3 py-2 rounded-lg text-sm font-bold ${
            recommendation.confidence > 0.7 ? 'bg-green-900/30 text-green-300' :
            recommendation.confidence > 0.4 ? 'bg-yellow-900/30 text-yellow-300' :
            'bg-red-900/30 text-red-300'
          }`}>
            {Math.round(recommendation.confidence * 100)}%
          </div>
        </div>
        
        <p className="text-base opacity-90 mb-3">{recommendation.reasoning}</p>
        
        {recommendation.hand_strength && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div className="bg-slate-900/50 p-2 rounded-lg">
              <div className="text-slate-400 text-xs mb-1">Force Main</div>
              <div className="font-semibold">{Math.round(recommendation.hand_strength * 100)}%</div>
            </div>
            {recommendation.pot_odds > 0 && (
              <div className="bg-slate-900/50 p-2 rounded-lg">
                <div className="text-slate-400 text-xs mb-1">Pot Odds</div>
                <div className="font-semibold">{recommendation.pot_odds.toFixed(1)}:1</div>
              </div>
            )}
            <div className="bg-slate-900/50 p-2 rounded-lg">
              <div className="text-slate-400 text-xs mb-1">Phase</div>
              <div className="font-semibold capitalize">{recommendation.phase}</div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Rendu des cartes détectées
  const renderDetectedCards = (cards, title) => {
    if (!cards || cards.length === 0) return null;

    return (
      <div className="bg-slate-900/50 p-4 rounded-lg">
        <div className="text-sm text-slate-400 mb-2">{title}</div>
        <div className="flex gap-2 flex-wrap">
          {cards.map((card, index) => (
            <div key={index} className="poker-card bg-white text-black px-3 py-2 rounded-lg font-bold text-lg">
              {card}
            </div>
          ))}
        </div>
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
                <p className="text-sm text-slate-400">
                  Analyse Ultra-Rapide • Texas Hold'em Spin & Go • v2.1
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Indicateur d'analyse en cours */}
              {isAnalyzing && (
                <div className="flex items-center gap-2 px-3 py-2 bg-blue-900/30 text-blue-300 rounded-lg">
                  <Loader className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Analyse...</span>
                </div>
              )}
              
              {/* Statut de connexion */}
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
                type="button"
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
                  <h2 className="text-lg font-semibold">Capture d'Écran Optimisée</h2>
                  {lastAnalysisTime && (
                    <span className="text-xs text-slate-400">
                      Dernière: {lastAnalysisTime.toLocaleTimeString()}
                    </span>
                  )}
                </div>
                <div className="flex gap-2">
                  {!isCapturing ? (
                    <button
                      onClick={startCapture}
                      className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors font-medium"
                      type="button"
                    >
                      <Play className="w-4 h-4" />
                      Démarrer
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={analyzeScreen}
                        disabled={isAnalyzing}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors font-medium ${
                          isAnalyzing 
                            ? 'bg-blue-400 cursor-not-allowed' 
                            : 'bg-blue-600 hover:bg-blue-700'
                        }`}
                        type="button"
                      >
                        {isAnalyzing ? <Loader className="w-4 h-4 animate-spin" /> : <Camera className="w-4 h-4" />}
                        {isAnalyzing ? 'Analyse...' : 'Analyser'}
                      </button>
                      <button
                        onClick={stopCapture}
                        className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors font-medium"
                        type="button"
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
                      <p className="text-xs text-slate-500 mt-2">Analyse optimisée &lt; 3s • Détection précise des cartes</p>
                    </div>
                  </div>
                )}
                <canvas ref={canvasRef} className="hidden" />
              </div>
            </div>

            {/* Résultats de l'analyse */}
            {currentAnalysis && !currentAnalysis.error && (
              <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
                <div className="flex items-center gap-3 mb-4">
                  <Target className="w-5 h-5 text-green-400" />
                  <h2 className="text-lg font-semibold">Analyse Détaillée</h2>
                  <div className="text-sm text-slate-400">
                    {new Date(currentAnalysis.timestamp).toLocaleTimeString()}
                  </div>
                  {currentAnalysis.processing_time && (
                    <div className={`text-sm font-bold ${
                      currentAnalysis.processing_time < 1 ? 'text-green-400' : 
                      currentAnalysis.processing_time < 3 ? 'text-blue-400' : 'text-yellow-400'
                    }`}>
                      ⚡ {currentAnalysis.processing_time.toFixed(2)}s
                    </div>
                  )}
                </div>
                
                {currentAnalysis.recommendation && renderRecommendation(currentAnalysis.recommendation)}
                
                {/* Cartes détectées */}
                <div className="mt-6 space-y-4">
                  {currentAnalysis.detected_elements?.hero_cards && 
                   renderDetectedCards(currentAnalysis.detected_elements.hero_cards, "Vos Cartes")}
                  
                  {currentAnalysis.detected_elements?.community_cards && 
                   renderDetectedCards(currentAnalysis.detected_elements.community_cards, "Board")}
                </div>
                
                {/* Informations détaillées */}
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                  {currentAnalysis.detected_elements?.pot && (
                    <div className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="text-sm text-slate-400 mb-1">Pot</div>
                      <div className="font-semibold">{currentAnalysis.detected_elements.pot}</div>
                    </div>
                  )}
                  
                  {currentAnalysis.detected_elements?.blinds && (
                    <div className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="text-sm text-slate-400 mb-1">Blinds</div>
                      <div className="font-semibold">
                        {currentAnalysis.detected_elements.blinds.small_blind || '?'}/
                        {currentAnalysis.detected_elements.blinds.big_blind || '?'}
                      </div>
                    </div>
                  )}
                  
                  {currentAnalysis.detected_elements?.betting_round && (
                    <div className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="text-sm text-slate-400 mb-1">Phase</div>
                      <div className="font-semibold capitalize">
                        {currentAnalysis.detected_elements.betting_round}
                      </div>
                    </div>
                  )}
                  
                  {currentAnalysis.confidence > 0 && (
                    <div className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="text-sm text-slate-400 mb-1">Confiance</div>
                      <div className="font-semibold">{Math.round(currentAnalysis.confidence * 100)}%</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Affichage d'erreur */}
            {currentAnalysis && currentAnalysis.error && (
              <div className="bg-red-900/20 border border-red-500/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-2">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                  <h3 className="text-lg font-semibold text-red-400">Erreur d'Analyse</h3>
                </div>
                <p className="text-red-300">{currentAnalysis.message}</p>
                <p className="text-xs text-red-400 mt-2">
                  Essayez de capturer une zone plus claire de la table de poker
                </p>
              </div>
            )}
          </div>

          {/* Panel latéral */}
          <div className="space-y-6">
            
            {/* Statistiques de performance */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-4">
                <Gauge className="w-5 h-5 text-purple-400" />
                <h2 className="text-lg font-semibold">Performance</h2>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-400">Analyses</span>
                  <span className="font-semibold">{stats.handsAnalyzed}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Confiance moy.</span>
                  <span className="font-semibold">{Math.round(stats.avgConfidence * 100)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Temps moy.</span>
                  <span className={`font-semibold ${
                    stats.avgProcessingTime < 1 ? 'text-green-400' : 
                    stats.avgProcessingTime < 3 ? 'text-blue-400' : 'text-yellow-400'
                  }`}>
                    {stats.avgProcessingTime.toFixed(1)}s
                  </span>
                </div>
                {stats.lastUpdateTime && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Dernière</span>
                    <span className="font-semibold text-sm">{stats.lastUpdateTime}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Historique */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-4">
                <Clock className="w-5 h-5 text-orange-400" />
                <h2 className="text-lg font-semibold">Historique</h2>
              </div>
              
              <div className="space-y-2 max-h-80 overflow-y-auto custom-scrollbar">
                {analysisHistory.length === 0 ? (
                  <p className="text-slate-400 text-sm">Aucune analyse encore</p>
                ) : (
                  analysisHistory.map((analysis, index) => (
                    <div key={analysis.id || index} className="bg-slate-900/50 p-3 rounded-lg">
                      <div className="flex items-center justify-between mb-1">
                        <span className={`text-sm font-medium ${
                          analysis.recommendation?.action === 'raise' ? 'text-green-400' :
                          analysis.recommendation?.action === 'call' ? 'text-yellow-400' :
                          analysis.recommendation?.action === 'fold' ? 'text-red-400' :
                          'text-slate-400'
                        }`}>
                          {analysis.recommendation?.action?.toUpperCase() || analysis.error ? 'ERREUR' : 'ANALYSE'}
                        </span>
                        <span className="text-xs text-slate-400">
                          {new Date(analysis.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>Confiance: {Math.round((analysis.confidence || 0) * 100)}%</span>
                        {analysis.processing_time && (
                          <span className={analysis.processing_time < 1 ? 'text-green-400' : 'text-blue-400'}>
                            ⚡ {analysis.processing_time.toFixed(1)}s
                          </span>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modal des paramètres simplifié */}
      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl p-6 max-w-md w-full mx-4 border border-slate-700">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold">Paramètres</h2>
              <button
                onClick={() => setIsSettingsOpen(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                type="button"
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
                  Fréquence d'analyse: {settings.captureFrequency}s
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={settings.captureFrequency}
                  onChange={(e) => setSettings({...settings, captureFrequency: parseInt(e.target.value)})}
                  className="w-full"
                />
                <p className="text-xs text-slate-400 mt-1">Optimisé pour 2-3s</p>
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
                  setIsSettingsOpen(false);
                  // Redémarrage avec nouveaux paramètres
                  if (isCapturing && intervalRef.current) {
                    clearInterval(intervalRef.current);
                    if (settings.autoAnalyze) {
                      intervalRef.current = setInterval(() => {
                        if (!isAnalyzing) {
                          analyzeScreen();
                        }
                      }, settings.captureFrequency * 1000);
                    }
                  }
                }}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors font-medium"
                type="button"
              >
                Sauvegarder
              </button>
              <button
                onClick={() => setIsSettingsOpen(false)}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                type="button"
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