"""
IrisFlow — Text-to-Speech (TTS)
Síntese de voz em português brasileiro.
Principal: Coqui TTS | Fallback: pyttsx3
"""

import os
from loguru import logger

TTS_ENGINE = os.getenv("TTS_ENGINE", "pyttsx3")


class TTSEngine:
    """
    Interface unificada de TTS.
    Troca o engine via variável de ambiente TTS_ENGINE.
    """

    def __init__(self) -> None:
        self._engine = TTS_ENGINE
        self._tts = None
        self._init_engine()

    def _init_engine(self) -> None:
        if self._engine == "coqui":
            self._init_coqui()
        else:
            self._init_pyttsx3()

    def _init_coqui(self) -> None:
        try:
            from TTS.api import TTS
            # Modelo PT-BR disponível no Coqui
            self._tts = TTS(model_name="tts_models/pt/cv/vits", progress_bar=False)
            logger.info("TTS iniciado: Coqui TTS (PT-BR)")
        except Exception as e:
            logger.warning("Coqui TTS falhou ({}), usando pyttsx3 como fallback.", e)
            self._engine = "pyttsx3"
            self._init_pyttsx3()

    def _init_pyttsx3(self) -> None:
        import pyttsx3
        self._tts = pyttsx3.init()
        # Configura voz em português, se disponível
        voices = self._tts.getProperty("voices")
        for voice in voices:
            if "brazil" in voice.id.lower() or "pt" in voice.id.lower():
                self._tts.setProperty("voice", voice.id)
                break
        self._tts.setProperty("rate", 150)   # Velocidade (palavras/minuto)
        self._tts.setProperty("volume", 1.0)
        logger.info("TTS iniciado: pyttsx3")

    def speak(self, text: str) -> None:
        """
        Sintetiza e reproduz o texto em voz alta.

        Args:
            text: Texto a ser falado em português.
        """
        if not text.strip():
            return

        logger.debug("TTS: '{}'", text)

        if self._engine == "coqui":
            self._tts.tts_to_file(text=text, file_path="/tmp/irisflow_tts.wav")
            os.system("aplay /tmp/irisflow_tts.wav 2>/dev/null || afplay /tmp/irisflow_tts.wav 2>/dev/null")
        else:
            self._tts.say(text)
            self._tts.runAndWait()

    def speak_async(self, text: str) -> None:
        """Versão assíncrona — fala sem bloquear o loop principal."""
        import threading
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()


# --- Teste rápido ---
if __name__ == "__main__":
    engine = TTSEngine()
    engine.speak("Olá! O IrisFlow está funcionando corretamente.")
