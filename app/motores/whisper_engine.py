import os
import sys
from typing import Dict, Any
from transformers import pipeline

# --- Configuración Preventiva y Portable de FFmpeg -------------------------
RUTAS_FFMPEG_POSIBLES = [
    r"C:\ffmpeg\bin",
    r"C:\ffmpeg",
    r"D:\ffmpeg\bin"
]

for _ruta in RUTAS_FFMPEG_POSIBLES:
    if os.path.exists(_ruta) and _ruta not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.path.pathsep + _ruta


class WhisperEngine:
    def __init__(self):
        print("⏳ [IA] Inicializando Motor 2: Transcriptor Whisper (openai/whisper-tiny)...")
        try:
            # Inicializamos el pipeline básico para Speech-to-Text
            self.pipe = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-tiny"
            )
            print("✅ [IA] Motor 2 (Whisper) cargado exitosamente.")
        except Exception as e:
            print(f"❌ [Whisper Error] No se pudo inicializar el modelo: {str(e)}")
            self.pipe = None

        # Variables de estado internas para almacenar la auditoría forense de la última llamada
        self.ultimo_idioma_detectado = "es"  # Por defecto forzado a español
        self.metricas_ultimo_analisis: Dict[str, Any] = {}

    def transcribir(self, ruta_audio: str) -> str:
        """
        Recibe la ruta absoluta de un archivo de audio, devuelve su transcripción 
        en texto y extrae métricas estructurales forenses en segundo plano.
        """
        if not os.path.exists(ruta_audio):
            raise FileNotFoundError(f"No se encontró el archivo de audio en: {ruta_audio}")
        
        if not self.pipe:
            print("⚠️ [Whisper] El pipeline no está operativo. Retornando cadena vacía.")
            return ""

        try:
            print(f"🎙️ [Whisper] Procesando señales de audio en: {os.path.basename(ruta_audio)}")
            
            # 💡 SOLUCIÓN DE PRECISIÓN SEMÁNTICA:
            # Forzamos 'generate_kwargs' con el idioma 'spanish' y la tarea 'transcribe'.
            # Esto evita que el modelo tiny "delire" o confunda fonemas locales con otros idiomas.
            resultado = self.pipe(
                ruta_audio, 
                return_timestamps=True,
                generate_kwargs={
                    "language": "spanish",
                    "task": "transcribe"
                }
            )
            
            texto_puro = resultado.get("text", "").strip()
            chunks_temporales = resultado.get("chunks", [])

            # Executamos el sub-análisis forense de metadatos de forma aislada
            self._extraer_metadatos_forenses(texto_puro, chunks_temporales)
            
            return texto_puro

        except Exception as e:
            print(f"❌ [Whisper] Error crítico durante la transcripción: {str(e)}")
            raise e

    def _extraer_metadatos_forenses(self, texto: str, chunks: list) -> None:
        """
        Método auxiliar privado para estructurar la metadata temporal del audio
        sin añadir retrasos ni llamadas extras a la arquitectura.
        """
        total_chunks = len(chunks)
        duracion_estimada = 0.0
        
        if total_chunks > 0 and chunks[-1].get("timestamp"):
            # Capturamos el segundo exacto en el que terminó de hablar la persona
            timestamp_final = chunks[-1]["timestamp"]
            if timestamp_final and len(timestamp_final) == 2:
                duracion_estimada = timestamp_final[1] if timestamp_final[1] else 0.0

        # Cálculo forense del ritmo del habla (Speech Rate) para detectar automatizaciones
        palabras = texto.split()
        total_palabras = len(palabras)
        palabras_por_segundo = round(total_palabras / duracion_estimada, 2) if duracion_estimada > 0 else 0.0

        # Almacenamos todo de forma segura en el estado de la clase
        self.metricas_ultimo_analisis = {
            "duracion_audio_segundos": round(duracion_estimada, 2),
            "total_fragmentos_deteccion": total_chunks,
            "densidad_habla_palabras_por_segundo": palabras_por_segundo,
            "segmentacion_lineal": chunks
        }

        # 📺 Print informativo en la terminal del desarrollador para validar el test local
        print("\n📊 --- AUDITORÍA INTERNA MOTORES (MI APARTADO WHISPER) ---")
        print(f"🌐 Idioma Forzado/Detectado: {self.ultimo_idioma_detectado.upper()}")
        print(f"⏱️ Duración Calculada: {duracion_estimada} segs")
        print(f"📈 Ritmo de Fluidez Semántica: {palabras_por_segundo} palabras/seg")
        print("----------------------------------------------------------\n")

    # --- Getters Públicos (Listos para cuando el Frontend decida llamarlos) ---
    def obtener_idioma_detectado(self) -> str:
        return self.ultimo_idioma_detectado

    def obtener_metricas_forenses(self) -> Dict[str, Any]:
        return self.metricas_ultimo_analisis


# Instancia única (Singleton) para no recargar el modelo en cada petición HTTP
whisper_engine = WhisperEngine()