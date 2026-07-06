import os
import sys
import wave
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
        # 🚀 ESCALADO A WHISPER-SMALL: Más capacidad conceptual para evitar fusiones acústicas erróneas
        # Tambien existen (base) que es un poco inferior a small y (tiny) que es más rápido pero menos preciso. Aquí priorizamos precisión.
        print("⏳ [IA] Inicializando Motor 2: Transcriptor Whisper Preciso (openai/whisper-small)...")
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-small",
                chunk_length_s=30,
                batch_size=8
            )
            print("✅ [IA] Motor 2 (Whisper Small) cargado exitosamente.")
        except Exception as e:
            print(f"❌ [Whisper Error] No se pudo inicializar el modelo Small: {str(e)}")
            self.pipe = None

        self.ultimo_idioma_detectado = "Detectando idioma nativo..."
        self.metricas_ultimo_analisis: Dict[str, Any] = {}

    def transcribir(self, ruta_audio: str) -> str:
        """
        Recibe la ruta absoluta de un archivo de audio, devuelve su transcripción 
        en texto y sincroniza el tiempo físico real usando librerías nativas.
        """
        if not os.path.exists(ruta_audio):
            raise FileNotFoundError(f"No se encontró el archivo de audio en: {ruta_audio}")
        
        if not self.pipe:
            print("⚠️ [Whisper] El pipeline no está operativo. Retornando cadena vacía.")
            return ""

        try:
            print(f"🎙️ [Whisper Small] Procesando señales de audio en: {os.path.basename(ruta_audio)}")
            
            # 🚀 DETECCIÓN Y TRANSCRIPCIÓN DINÁMICA:
            # Al no definir "language", Whisper detecta el idioma automáticamente en los primeros 30s 
            # y fuerza ese token de idioma internamente para mantener la máxima coherencia gramatical.
            resultado = self.pipe(
                ruta_audio, 
                return_timestamps=True,
                generate_kwargs={
                    "num_beams": 4,
                    "task": "transcribe"
                }
            )
            
            texto_puro = resultado.get("text", "").strip()
            chunks_temporales = resultado.get("chunks", [])

            # 🛠️ MEDICIÓN NATIVA DEL ARCHIVO REAL MEDIANTE METADATOS WAV
            duracion_fisica_real = 0.0
            try:
                with wave.open(ruta_audio, "rb") as archivo_wav:
                    frames = archivo_wav.getnframes()
                    rate = archivo_wav.getframerate()
                    duracion_fisica_real = frames / float(rate)
            except Exception:
                # Respaldo pasivo por si el archivo está corrupto o es de otro formato
                duracion_fisica_real = 0.0
            
            # 📊 Sincronizamos las métricas periciales cruzadas enviando los 3 parámetros requeridos
            self._extraer_metadatos_forenses(texto_puro, chunks_temporales, duracion_fisica_real)
            
            # 🚀 EL RETURN DEBE IR AL FINAL: Entregamos el texto limpio a main.py una vez calculado todo
            return texto_puro

        except Exception as e:
            print(f"❌ [Whisper Small] Error crítico durante la transcripción: {str(e)}")
            raise e

    def _extraer_metadatos_forenses(self, texto: str, chunks: list, duracion_fisica: float) -> None:
        """
        Cruza la información del archivo con la actividad de voz de la IA
        para calcular tasas de inactividad de forma segura y encapsulada.
        """
        total_chunks = len(chunks)
        duracion_habla = 0.0
        
        # Extraemos el segundo exacto donde terminó el último fragmento de voz legible
        if total_chunks > 0 and chunks[-1].get("timestamp"):
            timestamp_final = chunks[-1]["timestamp"]
            if timestamp_final and len(timestamp_final) == 2:
                duracion_habla = timestamp_final[1] if timestamp_final[1] else 0.0

        if duracion_fisica == 0.0:
            duracion_fisica = duracion_habla

        # 🔍 Métrica Forense Única: Segundos de silencio y pausas muertas
        tiempo_silencio = max(0.0, duracion_fisica - duracion_habla)
        porcentaje_silencio = round((tiempo_silencio / duracion_fisica) * 100, 2) if duracion_fisica > 0 else 0.0

        # Ritmo de palabras neta por segundo de habla activa
        palabras = texto.split()
        total_palabras = len(palabras)
        palabras_por_segundo = round(total_palabras / duracion_habla, 2) if duracion_habla > 0 else 0.0

        self.ultimo_idioma_detectado = "Inferencia Automática Whisper Small"

        # Guardamos todo en tu estado analítico interno
        self.metricas_ultimo_analisis = {
            "duracion_fisica_archivo_segundos": round(duracion_fisica, 2),
            "duracion_actividad_habla_segundos": round(duracion_habla, 2),
            "tiempo_inactividad_silencio_segundos": round(tiempo_silencio, 2),
            "porcentaje_silencio_llamada": porcentaje_silencio,
            "total_fragmentos_deteccion": total_chunks,
            "densidad_habla_palabras_por_segundo": palabras_por_segundo,
            "segmentacion_lineal": chunks
        }

        # 📺 Reporte pericial en consola de VS Code
        print("\n📊 --- AUDITORÍA DE ALTA PRECISIÓN (MÓDULO WHISPER FORENSE) ---")
        print(f"🌐 Idioma Detectado: {self.ultimo_idioma_detectado.upper()}")
        print(f"⏱️ Duración Física de Archivo: {round(duracion_fisica, 2)} segs")
        print(f"🎙️ Tiempo de Habla Efectiva (VAD): {round(duracion_habla, 2)} segs")
        print(f"🤫 Porcentaje de Silencio/Pausas: {porcentaje_silencio}%")
        print(f"📈 Ritmo del Habla: {palabras_por_segundo} palabras/seg")
        print("----------------------------------------------------------------\n")

    def obtener_idioma_detectado(self) -> str:
        return self.ultimo_idioma_detectado

    def obtener_metricas_forenses(self) -> Dict[str, Any]:
        return self.metricas_ultimo_analisis


# Instancia única (Singleton)
whisper_engine = WhisperEngine()