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
  Gauge,
  Terminal,
  Info
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
    showDebugLogs: true,
    sequentialMode: true // NOUVEAU : Mode séquentiel
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
  
  // NOUVEAU : États pour l'analyse séquentielle
  const [currentPhase, setCurrentPhase] = useState('preflop');
  const [phaseGuide, setPhaseGuide] = useState(null);
  const [detectedCards, setDetectedCards] = useState({
    hero: [],
    board: []
  });
  
  // NOUVEAU : Système de logs en temps réel
  const [debugLogs, setDebugLogs] = useState([]);
  const [currentStatus, setCurrentStatus] = useState('Prêt');
  const [analysisStep, setAnalysisStep] = useState('');
  
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  // NOUVEAU : Fonction pour ajouter des logs
  const addLog = useCallback((message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = {
      id: Date.now() + Math.random(),
      timestamp,
      message,
      type // 'info', 'success', 'error', 'warning'
    };
    
    console.log(`[${timestamp}] ${type.toUpperCase()}: ${message}`);
    
    setDebugLogs(prev => [newLog, ...prev.slice(0, 49)]); // Garde 50 logs max
  }, []);

  // NOUVEAU : Fonction pour mettre à jour le statut
  const updateStatus = useCallback((status, step = '') => {
    setCurrentStatus(status);
    setAnalysisStep(step);
    addLog(`Status: ${status}${step ? ` - ${step}` : ''}`, 'info');
  }, [addLog]);

  // Initialisation
  useEffect(() => {
    addLog('🚀 Assistant Poker Pro v2.1 initialisé', 'success');
    addLog(`📡 Backend URL: ${BACKEND_URL}`, 'info');
    addLog(`🔗 Session ID: ${sessionId}`, 'info');
    updateStatus('Prêt', 'En attente de capture d\'écran');
  }, [addLog, updateStatus, sessionId]);

  // Connexion WebSocket avec logs
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        addLog('🔌 Tentative de connexion WebSocket...', 'info');
        wsRef.current = new WebSocket(`${WS_URL}/${sessionId}`);
        
        wsRef.current.onopen = () => {
          setConnectionStatus('connected');
          addLog('✅ WebSocket connecté avec succès', 'success');
          updateStatus('Connecté', 'WebSocket actif');
        };
        
        wsRef.current.onmessage = (event) => {
          addLog('📨 Message WebSocket reçu', 'info');
          try {
            const message = JSON.parse(event.data);
            addLog(`📋 Type de message: ${message.type}`, 'info');
            if (message.type === 'analysis_result') {
              addLog('🎯 Résultat d\'analyse reçu via WebSocket', 'success');
              handleAnalysisResult(message.data);
            }
          } catch (error) {
            addLog(`❌ Erreur parsing WebSocket: ${error.message}`, 'error');
          }
        };
        
        wsRef.current.onclose = () => {
          setConnectionStatus('disconnected');
          addLog('🔌 WebSocket déconnecté, reconnexion dans 3s...', 'warning');
          updateStatus('Déconnecté', 'Reconnexion automatique');
          setTimeout(connectWebSocket, 3000);
        };
        
        wsRef.current.onerror = (error) => {
          setConnectionStatus('error');
          addLog(`❌ Erreur WebSocket: ${error}`, 'error');
          updateStatus('Erreur WebSocket');
        };
      } catch (error) {
        addLog(`❌ Erreur connexion WebSocket: ${error.message}`, 'error');
        setConnectionStatus('error');
      }
    };

    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        addLog('🔌 Fermeture WebSocket', 'info');
        wsRef.current.close();
      }
    };
  }, [sessionId, addLog, updateStatus]);

  // Gestionnaire de résultats d'analyse avec logs détaillés
  const handleAnalysisResult = useCallback((analysisData) => {
    addLog('📊 Traitement du résultat d\'analyse...', 'info');
    
    setCurrentAnalysis(analysisData);
    setAnalysisHistory(prev => [analysisData, ...prev.slice(0, 19)]);
    
    // Logs détaillés sur le résultat
    if (analysisData.error) {
      addLog(`❌ Erreur d'analyse: ${analysisData.message}`, 'error');
    } else {
      addLog(`✅ Analyse réussie en ${analysisData.processing_time?.toFixed(2)}s`, 'success');
      
      if (analysisData.detected_elements?.hero_cards) {
        addLog(`🃏 Cartes détectées: ${analysisData.detected_elements.hero_cards.join(', ')}`, 'success');
      }
      
      if (analysisData.detected_elements?.community_cards?.length > 0) {
        addLog(`🎰 Board: ${analysisData.detected_elements.community_cards.join(', ')}`, 'success');
      }
      
      if (analysisData.recommendation) {
        addLog(`💡 Recommandation: ${analysisData.recommendation.action?.toUpperCase()} (${Math.round(analysisData.recommendation.confidence * 100)}%)`, 'success');
      }
    }
    
    // Mise à jour des statistiques avec logs
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

      addLog(`📈 Stats mises à jour: ${newCount} analyses, ${Math.round(newAvgConfidence * 100)}% confiance moy.`, 'info');

      return {
        handsAnalyzed: newCount,
        avgConfidence: newAvgConfidence,
        lastUpdateTime: new Date().toLocaleTimeString(),
        avgProcessingTime: newAvgTime
      };
    });
    
    setIsAnalyzing(false);
    setLastAnalysisTime(new Date());
    updateStatus('Analyse terminée', `${analysisData.processing_time?.toFixed(2)}s`);
  }, [addLog, updateStatus]);

  // NOUVELLE : Fonction d'analyse avec détection de phase
  const analyzeScreenWithPhase = useCallback(async (phaseHint = null) => {
    if (!stream || !videoRef.current || !canvasRef.current) {
      addLog('❌ Conditions non remplies pour l\'analyse', 'error');
      addLog(`Stream: ${!!stream}, Video: ${!!videoRef.current}, Canvas: ${!!canvasRef.current}`, 'error');
      return;
    }

    if (isAnalyzing) {
      addLog('⏳ Analyse déjà en cours, ignore la nouvelle demande', 'warning');
      return;
    }

    const phase = phaseHint || currentPhase;
    addLog(`🚀 ANALYSE ${phase.toUpperCase()} EN COURS`, 'info');
    setIsAnalyzing(true);
    setLastAnalysisTime(new Date());
    updateStatus(`Analyse ${phase}`, 'Capture de l\'image...');

    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    try {
      addLog(`📷 Capture pour phase ${phase}...`, 'info');
      const ctx = canvas.getContext('2d');
      
      // Optimisation : résolution adaptée à la phase MAIS PLUS PETITE
      const targetWidth = 640;   // Taille fixe ultra-optimisée pour vitesse
      const targetHeight = 360;  // Taille fixe ultra-optimisée pour vitesse
      
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
      
      addLog(`✅ Image capturée: ${targetWidth}x${targetHeight} pour ${phase}`, 'success');
      updateStatus(`Analyse ${phase}`, 'Conversion de l\'image...');
      
      // Conversion optimisée en base64 ULTRA-RAPIDE
      const imageData = canvas.toDataURL('image/jpeg', 0.70); // Qualité réduite pour vitesse max
      const base64Data = imageData.split(',')[1];
      
      addLog(`📦 Image convertie: ${(base64Data.length / 1024).toFixed(1)}KB`, 'success');
      updateStatus(`Analyse ${phase}`, `Envoi à l\'API avec focus ${phase}...`);
      
      // Préparation de la requête avec hint de phase
      const requestData = {
        image_base64: base64Data,
        session_id: sessionId,
        phase_hint: phase // NOUVEAU : Hint pour l'analyse spécialisée
      };
      
      addLog(`📡 Envoi requête avec phase_hint: ${phase}`, 'info');
      
      const startTime = Date.now();
      
      const response = await fetch(`${API}/analyze-screen`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });
      
      const responseTime = Date.now() - startTime;
      addLog(`📡 Réponse reçue en ${responseTime}ms`, 'info');
      addLog(`📊 Status HTTP: ${response.status} ${response.statusText}`, response.ok ? 'success' : 'error');
      
      updateStatus(`Analyse ${phase}`, 'Traitement de la réponse...');
      
      if (response.ok) {
        const result = await response.json();
        addLog('✅ Réponse JSON parsée avec succès', 'success');
        
        // Log spécifique selon la phase
        if (result.detected_elements) {
          const heroCards = result.detected_elements.hero_cards || [];
          const boardCards = result.detected_elements.community_cards || [];
          
          addLog(`🃏 Cartes héros détectées: ${heroCards.length ? heroCards.join(', ') : 'Aucune'}`, heroCards.length ? 'success' : 'warning');
          addLog(`🎯 Board détecté: ${boardCards.length ? boardCards.join(', ') : 'Aucun'}`, boardCards.length || phase === 'preflop' ? 'success' : 'warning');
          
          // Mise à jour des cartes détectées
          setDetectedCards({
            hero: heroCards,
            board: boardCards
          });
          
          // Validation de la phase
          const detectedPhase = result.detected_elements.betting_round;
          if (detectedPhase && detectedPhase !== phase) {
            addLog(`🔄 Phase détectée différente: ${phase} → ${detectedPhase}`, 'info');
            setCurrentPhase(detectedPhase);
          }
        }
        
        if (!result.error) {
          addLog(`🎯 Analyse ${phase} réussie !`, 'success');
          
          // Log de la recommandation
          if (result.recommendation) {
            const rec = result.recommendation;
            addLog(`💡 RECOMMANDATION: ${rec.action?.toUpperCase()} (${Math.round(rec.confidence * 100)}%)`, 'success');
            if (rec.equity) {
              addLog(`⚡ Équité: ${Math.round(rec.equity * 100)}% | Type: ${rec.hand_type || 'N/A'}`, 'info');
            }
            if (rec.outs > 0) {
              addLog(`🎲 Outs: ${rec.outs} | Pot odds: ${rec.pot_odds?.toFixed(1)}:1`, 'info');
            }
          }
          
          handleAnalysisResult(result);
        } else {
          addLog(`❌ Erreur API: ${result.message}`, 'error');
          setIsAnalyzing(false);
          updateStatus('Erreur', result.message);
        }
      } else {
        const errorText = await response.text();
        addLog(`❌ Erreur HTTP ${response.status}: ${errorText}`, 'error');
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
    } catch (error) {
      addLog(`❌ ERREUR ${phase.toUpperCase()}: ${error.message}`, 'error');
      setIsAnalyzing(false);
      updateStatus('Erreur critique', error.message);
      
      // Affichage d'erreur temporaire
      setCurrentAnalysis({
        error: true,
        message: `Erreur analyse ${phase}: ${error.message}`,
        timestamp: new Date().toISOString()
      });
      
      setTimeout(() => {
        setCurrentAnalysis(null);
        updateStatus('Prêt', 'Erreur résolue');
      }, 10000);
    }
  }, [stream, sessionId, isAnalyzing, currentPhase, handleAnalysisResult, addLog, updateStatus]);

  // Fonction d'analyse standard (fallback)
  const analyzeScreen = useCallback(() => {
    return analyzeScreenWithPhase();
  }, [analyzeScreenWithPhase]);

  // Démarrage de la capture avec logs
  const startCapture = async () => {
    try {
      addLog('📺 Demande de capture d\'écran...', 'info');
      updateStatus('Capture en cours', 'Demande de permissions...');
      
      const mediaStream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          mediaSource: 'screen',
          width: { ideal: 1920, max: 1920 },
          height: { ideal: 1080, max: 1080 },
          frameRate: { ideal: 30, max: 30 }
        },
        audio: false
      });
      
      addLog('✅ Permissions accordées, stream obtenu', 'success');
      addLog(`📊 Stream: ${mediaStream.getVideoTracks().length} pistes vidéo`, 'info');
      
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        addLog('📹 Stream assigné à l\'élément vidéo', 'success');
      }
      
      setIsCapturing(true);
      updateStatus('Capture active', 'Prêt pour analyse');
      
      // Capture automatique selon paramètres
      if (settings.autoAnalyze) {
        const frequency = settings.captureFrequency * 1000;
        addLog(`⏰ Analyse automatique activée: ${settings.captureFrequency}s`, 'info');
        
        intervalRef.current = setInterval(() => {
          if (!isAnalyzing) {
            addLog('🔄 Déclenchement analyse automatique', 'info');
            analyzeScreen();
          } else {
            addLog('⏳ Analyse en cours, skip intervalle', 'warning');
          }
        }, frequency);
      }
      
    } catch (error) {
      addLog(`❌ Erreur capture: ${error.message}`, 'error');
      updateStatus('Erreur capture', error.message);
      alert('Impossible d\'accéder à la capture d\'écran. Vérifiez les autorisations.');
    }
  };

  // Arrêt de la capture avec logs
  const stopCapture = () => {
    addLog('🛑 Arrêt de la capture...', 'info');
    
    if (stream) {
      stream.getTracks().forEach(track => {
        track.stop();
        addLog(`🔌 Piste ${track.kind} fermée`, 'info');
      });
      setStream(null);
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      addLog('⏰ Intervalle automatique supprimé', 'info');
    }
    
    setIsCapturing(false);
    setIsAnalyzing(false);
    updateStatus('Prêt', 'Capture arrêtée');
    addLog('✅ Capture arrêtée avec succès', 'success');
  };

  // Nettoyage avec logs
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        addLog('🧹 Nettoyage: intervalle supprimé', 'info');
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        addLog('🧹 Nettoyage: stream fermé', 'info');
      }
    };
  }, [stream, addLog]);

  // Rendu détaillé de la recommandation avec BEAUCOUP de statistiques
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

    // Calculs de statistiques supplémentaires
    const handStrengthPercent = Math.round((recommendation.hand_strength || 0) * 100);
    const confidencePercent = Math.round(recommendation.confidence * 100);
    
    // Évaluation de la force de la décision
    const getDecisionStrength = () => {
      if (confidencePercent >= 80) return { text: 'TRÈS FORTE', color: 'text-green-400' };
      if (confidencePercent >= 60) return { text: 'FORTE', color: 'text-green-300' };
      if (confidencePercent >= 40) return { text: 'MODÉRÉE', color: 'text-yellow-400' };
      return { text: 'FAIBLE', color: 'text-red-400' };
    };

    const decisionStrength = getDecisionStrength();

    return (
      <div className={`p-6 rounded-xl border-2 ${actionColors[recommendation.action] || 'text-gray-400 bg-gray-900/20 border-gray-500/30'} recommendation-card`}>
        {/* En-tête de la recommandation */}
        <div className="flex items-center gap-4 mb-4">
          {actionIcons[recommendation.action]}
          <div className="flex-1">
            <h3 className="text-3xl font-bold mb-1">
              {actionLabels[recommendation.action] || recommendation.action?.toUpperCase()}
            </h3>
            <div className="flex items-center gap-3">
              <span className={`text-sm font-bold ${decisionStrength.color}`}>
                🎯 DÉCISION {decisionStrength.text}
              </span>
              <span className="text-sm text-slate-400">
                Phase: {recommendation.phase || 'Inconnue'}
              </span>
            </div>
          </div>
          <div className={`px-4 py-3 rounded-xl text-lg font-bold ${
            confidencePercent >= 80 ? 'bg-green-900/40 text-green-300' :
            confidencePercent >= 60 ? 'bg-blue-900/40 text-blue-300' :
            confidencePercent >= 40 ? 'bg-yellow-900/40 text-yellow-300' :
            'bg-red-900/40 text-red-300'
          }`}>
            {confidencePercent}%
          </div>
        </div>
        
        {/* Raisonnement détaillé */}
        <div className="bg-slate-900/30 p-4 rounded-lg mb-4">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">🧠 ANALYSE STRATÉGIQUE</h4>
          <p className="text-base text-slate-100">{recommendation.reasoning}</p>
        </div>
        
        {/* Statistiques détaillées */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          {/* Force de main */}
          <div className="bg-slate-900/50 p-4 rounded-lg text-center">
            <div className="text-slate-400 text-xs mb-1">FORCE MAIN</div>
            <div className={`text-2xl font-bold ${
              handStrengthPercent >= 80 ? 'text-green-400' :
              handStrengthPercent >= 60 ? 'text-blue-400' :
              handStrengthPercent >= 40 ? 'text-yellow-400' :
              'text-red-400'
            }`}>
              {handStrengthPercent}%
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {handStrengthPercent >= 80 ? 'Excellente' :
               handStrengthPercent >= 60 ? 'Bonne' :
               handStrengthPercent >= 40 ? 'Moyenne' : 'Faible'}
            </div>
          </div>

          {/* Pot Odds */}
          {recommendation.pot_odds > 0 && (
            <div className="bg-slate-900/50 p-4 rounded-lg text-center">
              <div className="text-slate-400 text-xs mb-1">POT ODDS</div>
              <div className="text-2xl font-bold text-purple-400">
                {recommendation.pot_odds.toFixed(1)}:1
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {recommendation.pot_odds >= 3 ? 'Favorables' :
                 recommendation.pot_odds >= 2 ? 'Correctes' : 'Défavorables'}
              </div>
            </div>
          )}

          {/* Équité estimée */}
          <div className="bg-slate-900/50 p-4 rounded-lg text-center">
            <div className="text-slate-400 text-xs mb-1">ÉQUITÉ</div>
            <div className={`text-2xl font-bold ${
              handStrengthPercent >= 60 ? 'text-green-400' :
              handStrengthPercent >= 40 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {handStrengthPercent}%
            </div>
            <div className="text-xs text-slate-400 mt-1">vs adversaires</div>
          </div>

          {/* Position */}
          <div className="bg-slate-900/50 p-4 rounded-lg text-center">
            <div className="text-slate-400 text-xs mb-1">POSITION</div>
            <div className="text-lg font-bold text-blue-400">
              {recommendation.phase === 'heads_up' ? 'HU' :
               recommendation.phase === 'late' ? 'LATE' :
               recommendation.phase === 'middle' ? 'MID' : 'EARLY'}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {recommendation.phase === 'heads_up' ? 'Heads-Up' :
               recommendation.phase === 'late' ? 'Fin de partie' :
               recommendation.phase === 'middle' ? 'Milieu' : 'Début'}
            </div>
          </div>
        </div>

        {/* Statistiques avancées */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* Analyse de risque */}
          <div className="bg-slate-900/30 p-4 rounded-lg">
            <h5 className="text-sm font-semibold text-slate-300 mb-3">⚖️ ANALYSE DE RISQUE</h5>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Risque/Récompense:</span>
                <span className={`font-semibold ${
                  recommendation.action === 'raise' ? 'text-green-400' :
                  recommendation.action === 'call' ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {recommendation.action === 'raise' ? 'Favorable' :
                   recommendation.action === 'call' ? 'Neutre' : 'Évité'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Niveau agressivité:</span>
                <span className="font-semibold text-blue-400">
                  {recommendation.action === 'raise' || recommendation.action === 'all_in' ? 'Élevé' :
                   recommendation.action === 'call' ? 'Modéré' : 'Faible'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Potentiel bluff:</span>
                <span className="font-semibold text-purple-400">
                  {handStrengthPercent < 30 && recommendation.action === 'raise' ? 'Possible' : 'Peu probable'}
                </span>
              </div>
            </div>
          </div>

          {/* Facteurs décisionnels */}
          <div className="bg-slate-900/30 p-4 rounded-lg">
            <h5 className="text-sm font-semibold text-slate-300 mb-3">🎲 FACTEURS CLÉS</h5>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Stack pressure:</span>
                <span className={`font-semibold ${
                  recommendation.phase === 'late' ? 'text-red-400' :
                  recommendation.phase === 'middle' ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {recommendation.phase === 'late' ? 'Élevée' :
                   recommendation.phase === 'middle' ? 'Modérée' : 'Faible'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">ICM pressure:</span>
                <span className="font-semibold text-orange-400">
                  {recommendation.phase === 'heads_up' ? 'Critique' :
                   recommendation.phase === 'late' ? 'Élevée' : 'Normale'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Fold equity:</span>
                <span className="font-semibold text-cyan-400">
                  {recommendation.action === 'raise' ? 'Bonne' :
                   recommendation.action === 'call' ? 'Limitée' : 'Nulle'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Plan d'action détaillé */}
        <div className="bg-gradient-to-r from-slate-900/50 to-slate-800/50 p-4 rounded-lg">
          <h5 className="text-sm font-semibold text-slate-300 mb-3">📋 PLAN D'ACTION</h5>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-slate-400">Action principale:</span>
              <div className={`font-bold text-lg ${
                recommendation.action === 'raise' ? 'text-green-400' :
                recommendation.action === 'call' ? 'text-yellow-400' :
                recommendation.action === 'fold' ? 'text-red-400' : 'text-purple-400'
              }`}>
                {actionLabels[recommendation.action] || recommendation.action?.toUpperCase()}
              </div>
            </div>
            <div>
              <span className="text-slate-400">Alternative:</span>
              <div className="font-medium text-slate-300">
                {recommendation.action === 'raise' ? 'Call acceptable' :
                 recommendation.action === 'call' ? 'Fold si pressure' :
                 recommendation.action === 'fold' ? 'Pas d\'alternative' : 'Raise si bluff'}
              </div>
            </div>
            <div>
              <span className="text-slate-400">Timing:</span>
              <div className="font-medium text-blue-400">
                {recommendation.action === 'fold' ? 'Immédiat' :
                 recommendation.action === 'call' ? 'Réfléchi' : 'Rapide'}
              </div>
            </div>
          </div>
        </div>

        {/* Barre de confiance visuelle */}
        <div className="mt-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-slate-400">Niveau de confiance</span>
            <span className="text-sm font-semibold">{confidencePercent}%</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all duration-1000 ${
                confidencePercent >= 80 ? 'bg-gradient-to-r from-green-500 to-green-400' :
                confidencePercent >= 60 ? 'bg-gradient-to-r from-blue-500 to-blue-400' :
                confidencePercent >= 40 ? 'bg-gradient-to-r from-yellow-500 to-yellow-400' :
                'bg-gradient-to-r from-red-500 to-red-400'
              }`}
              style={{width: `${confidencePercent}%`}}
            />
          </div>
        </div>
      </div>
    );
  };

  // Rendu des cartes détectées (inchangé)
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
      {/* Header avec statut */}
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
                  🎯 Analyse OCR Avancée • v4.1 • {currentStatus} {analysisStep && `• ${analysisStep}`}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Indicateur d'analyse détaillé */}
              {isAnalyzing && (
                <div className="flex items-center gap-2 px-3 py-2 bg-blue-900/30 text-blue-300 rounded-lg">
                  <Loader className="w-4 h-4 animate-spin" />
                  <span className="text-sm">{analysisStep || 'Analyse...'}</span>
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
            
            {/* NOUVEAU : Guide d'analyse séquentielle */}
            {settings.sequentialMode && (
              <div className="bg-gradient-to-r from-purple-900/20 to-blue-900/20 border border-purple-500/30 rounded-xl p-6 mb-6">
                <div className="flex items-center gap-3 mb-4">
                  <Target className="w-5 h-5 text-purple-400" />
                  <h2 className="text-lg font-semibold">Guide d'Analyse Séquentielle</h2>
                  <span className="px-2 py-1 bg-purple-900/30 text-purple-300 text-xs rounded-lg">
                    Phase: {currentPhase.toUpperCase()}
                  </span>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                  {/* Phases de jeu */}
                  {['preflop', 'flop', 'turn', 'river'].map((phase, index) => (
                    <div 
                      key={phase}
                      className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                        currentPhase === phase 
                          ? 'border-purple-400 bg-purple-900/30' 
                          : 'border-slate-600 bg-slate-800/50 hover:border-slate-500'
                      }`}
                      onClick={() => setCurrentPhase(phase)}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div className={`w-3 h-3 rounded-full ${
                          currentPhase === phase ? 'bg-purple-400' : 'bg-slate-500'
                        }`} />
                        <span className="text-sm font-semibold capitalize">{phase}</span>
                      </div>
                      <div className="text-xs text-slate-400">
                        {phase === 'preflop' ? '0 cartes board' :
                         phase === 'flop' ? '3 cartes board' :
                         phase === 'turn' ? '4 cartes board' : '5 cartes board'}
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Instructions selon la phase */}
                <div className="bg-slate-900/30 p-4 rounded-lg">
                  <h4 className="text-sm font-semibold text-purple-300 mb-2">
                    📋 Instructions pour {currentPhase.toUpperCase()}
                  </h4>
                  <p className="text-sm text-slate-300">
                    {currentPhase === 'preflop' ? 
                      '🃏 Montrez vos 2 cartes personnelles. Pas de cartes communes visibles.' :
                     currentPhase === 'flop' ?
                      '🎯 Montrez vos cartes + exactement 3 cartes communes au centre (le flop).' :
                     currentPhase === 'turn' ?
                      '🎲 Montrez vos cartes + 4 cartes communes (flop + turn card).' :
                      '🏁 Montrez vos cartes + toutes les 5 cartes communes (board complet).'}
                  </p>
                  
                  {/* Bouton d'analyse spécialisée */}
                  <button
                    onClick={() => {
                      addLog(`👆 Analyse ${currentPhase} spécialisée demandée`, 'info');
                      analyzeScreenWithPhase(currentPhase);
                    }}
                    disabled={!isCapturing || isAnalyzing}
                    className={`mt-3 flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                      !isCapturing || isAnalyzing
                        ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                        : 'bg-purple-600 hover:bg-purple-700 text-white'
                    }`}
                    type="button"
                  >
                    {isAnalyzing ? <Loader className="w-4 h-4 animate-spin" /> : <Camera className="w-4 h-4" />}
                    Analyser {currentPhase.toUpperCase()}
                  </button>
                </div>
                
                {/* Affichage des cartes détectées */}
                {(detectedCards.hero.length > 0 || detectedCards.board.length > 0) && (
                  <div className="mt-4 bg-slate-900/30 p-4 rounded-lg">
                    <h4 className="text-sm font-semibold text-green-300 mb-3">🎯 Cartes Détectées</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {detectedCards.hero.length > 0 && (
                        <div>
                          <div className="text-xs text-slate-400 mb-2">Vos cartes :</div>
                          <div className="flex gap-2">
                            {detectedCards.hero.map((card, index) => (
                              <div key={index} className="bg-white text-black px-2 py-1 rounded text-sm font-bold">
                                {card}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {detectedCards.board.length > 0 && (
                        <div>
                          <div className="text-xs text-slate-400 mb-2">Board ({detectedCards.board.length} cartes) :</div>
                          <div className="flex gap-2">
                            {detectedCards.board.map((card, index) => (
                              <div key={index} className="bg-yellow-100 text-black px-2 py-1 rounded text-sm font-bold">
                                {card}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Contrôles de capture avec debug */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Eye className="w-5 h-5 text-blue-400" />
                  <h2 className="text-lg font-semibold">Capture Debug Mode</h2>
                  {lastAnalysisTime && (
                    <span className="text-xs text-slate-400">
                      {lastAnalysisTime.toLocaleTimeString()}
                    </span>
                  )}
                </div>
                <div className="flex gap-2">
                  {!isCapturing ? (
                    <button
                      onClick={() => {
                        addLog('👆 Bouton "Démarrer" cliqué', 'info');
                        startCapture();
                      }}
                      className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors font-medium"
                      type="button"
                    >
                      <Play className="w-4 h-4" />
                      Démarrer
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={() => {
                          addLog('👆 Bouton "Analyser" cliqué', 'info');
                          analyzeScreen();
                        }}
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
                        onClick={() => {
                          addLog('👆 Bouton "Arrêter" cliqué', 'info');
                          stopCapture();
                        }}
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
              
              {/* Statut en temps réel */}
              <div className="mb-4 p-3 bg-slate-900/30 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Info className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-medium">Statut: {currentStatus}</span>
                </div>
                {analysisStep && (
                  <div className="text-xs text-slate-400">➤ {analysisStep}</div>
                )}
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
                      <p className="text-slate-400">Mode Debug Activé</p>
                      <p className="text-xs text-slate-500 mt-2">Tous les logs sont visibles en temps réel</p>
                    </div>
                  </div>
                )}
                <canvas ref={canvasRef} className="hidden" />
              </div>
            </div>

            {/* Résultats de l'analyse (inchangé) */}
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
              </div>
            )}
          </div>

          {/* Panel latéral avec LOGS DEBUG */}
          <div className="space-y-6">
            
            {/* NOUVEAU: Console de logs en temps réel */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-4">
                <Terminal className="w-5 h-5 text-green-400" />
                <h2 className="text-lg font-semibold">Console Debug</h2>
                <button
                  onClick={() => {
                    setDebugLogs([]);
                    addLog('🧹 Console nettoyée', 'info');
                  }}
                  className="text-xs px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded transition-colors"
                  type="button"
                >
                  Clear
                </button>
              </div>
              
              <div className="bg-black/50 p-3 rounded-lg h-60 overflow-y-auto font-mono text-xs custom-scrollbar">
                {debugLogs.length === 0 ? (
                  <div className="text-slate-500">En attente de logs...</div>
                ) : (
                  debugLogs.map((log) => (
                    <div key={log.id} className={`mb-1 ${
                      log.type === 'error' ? 'text-red-400' :
                      log.type === 'success' ? 'text-green-400' :
                      log.type === 'warning' ? 'text-yellow-400' :
                      'text-slate-300'
                    }`}>
                      <span className="text-slate-500">[{log.timestamp}]</span> {log.message}
                    </div>
                  ))
                )}
              </div>
            </div>
            
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
              
              <div className="space-y-2 max-h-60 overflow-y-auto custom-scrollbar">
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

      {/* Modal des paramètres avec debug */}
      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl p-6 max-w-md w-full mx-4 border border-slate-700">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold">Paramètres Debug</h2>
              <button
                onClick={() => {
                  addLog('👆 Modal paramètres fermé', 'info');
                  setIsSettingsOpen(false);
                }}
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
                  onChange={(e) => {
                    const value = parseFloat(e.target.value);
                    setSettings({...settings, aggressiveness: value});
                    addLog(`⚙️ Agressivité changée: ${Math.round(value * 100)}%`, 'info');
                  }}
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
                  onChange={(e) => {
                    const value = parseInt(e.target.value);
                    setSettings({...settings, captureFrequency: value});
                    addLog(`⚙️ Fréquence changée: ${value}s`, 'info');
                  }}
                  className="w-full"
                />
                <p className="text-xs text-slate-400 mt-1">Optimisé pour 2-3s</p>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Analyse automatique</span>
                <input
                  type="checkbox"
                  checked={settings.autoAnalyze}
                  onChange={(e) => {
                    const value = e.target.checked;
                    setSettings({...settings, autoAnalyze: value});
                    addLog(`⚙️ Analyse auto: ${value ? 'ON' : 'OFF'}`, 'info');
                  }}
                  className="w-4 h-4"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Mode séquentiel</span>
                <input
                  type="checkbox"
                  checked={settings.sequentialMode}
                  onChange={(e) => {
                    const value = e.target.checked;
                    setSettings({...settings, sequentialMode: value});
                    addLog(`⚙️ Mode séquentiel: ${value ? 'ON' : 'OFF'}`, value ? 'success' : 'info');
                  }}
                  className="w-4 h-4"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Debug logs</span>
                <input
                  type="checkbox"
                  checked={settings.showDebugLogs}
                  onChange={(e) => {
                    const value = e.target.checked;
                    setSettings({...settings, showDebugLogs: value});
                    addLog(`⚙️ Debug logs: ${value ? 'ON' : 'OFF'}`, value ? 'success' : 'warning');
                  }}
                  className="w-4 h-4"
                />
              </div>
            </div>
            
            <div className="bg-slate-900/50 p-3 rounded-lg mt-4">
              <div className="text-xs text-slate-400">
                <strong>Mode Séquentiel:</strong> Guide phase par phase (preflop → flop → turn → river)<br/>
                <strong>Debug Logs:</strong> Affiche tous les logs d'analyse en temps réel
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  addLog('💾 Paramètres sauvegardés', 'success');
                  setIsSettingsOpen(false);
                  // Redémarrage avec nouveaux paramètres
                  if (isCapturing && intervalRef.current) {
                    clearInterval(intervalRef.current);
                    if (settings.autoAnalyze) {
                      intervalRef.current = setInterval(() => {
                        if (!isAnalyzing) {
                          addLog('🔄 Auto-analyse déclenchée', 'info');
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
                onClick={() => {
                  addLog('❌ Paramètres annulés', 'warning');
                  setIsSettingsOpen(false);
                }}
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