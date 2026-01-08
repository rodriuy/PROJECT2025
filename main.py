import os
import sys
import json
import time
import tempfile
import threading
import queue
import subprocess
import traceback
from pathlib import Path
import shlex
import shutil
import importlib
import sys as _sys
import subprocess as _subprocess

# helper to attempt pip installs when auto_install_deps is enabled
def try_install(package):
    try:
        dbg(f"Intentando pip install {package}")
        _subprocess.check_call([_sys.executable, "-m", "pip", "install", "--user", package],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        importlib.invalidate_caches()
        dbg(f"pip install {package} OK")
        return True
    except Exception as e:
        dbg(f"pip install {package} falló: {repr(e)}")
        return False

# Cargar configuración
BASE_DIR = Path(__file__).resolve().parent
CFG_PATH = BASE_DIR / "config.json"
if not CFG_PATH.exists():
    print("Falta config.json. Crear según instrucciones en encabezado.", file=sys.stderr)
    sys.exit(1)
with CFG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# Utilidades ligeras
def safe_path(p):
    # Si no hay valor, devolver cadena vacía
    if not p:
        return ""
    # Si ya tiene esquema explícito, devolver tal cual
    if isinstance(p, str) and (p.startswith("http://") or p.startswith("https://")):
        return p
    # Si parece una URL con host/path (ej: domain.tld/...), anteponer https://
    if isinstance(p, str) and "/" in p and not p.startswith("./") and not p.startswith("../"):
        first = p.split("/")[0]
        if "." in first:
            # tratar como URL pública
            return "https://" + p if not p.startswith("http") else p
    # Por defecto: tratar como ruta local (resolver contra BASE_DIR si no es absoluta)
    if isinstance(p, str):
        try:
            pp = Path(p)
            return str(pp.resolve()) if pp.is_absolute() else str((BASE_DIR / p).resolve())
        except Exception:
            return str(p)
    # fallback
    return str(p)

def dbg(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr)

# Cargar persona por defecto
PERSON_KEY = next(iter(CONFIG.get("kelly") and {"kelly"} or CONFIG.get("persons", {}).keys()))
PERSON = CONFIG.get("kelly") or CONFIG.get("persons", {}).get(PERSON_KEY)
PERSON["photo"] = safe_path(PERSON.get("photo"))
PERSON["idle_video"] = safe_path(PERSON.get("idle_video"))
PERSON["planb_video"] = safe_path(PERSON.get("planb_video"))
PERSON["context_prompt_file"] = safe_path(PERSON.get("context_prompt_file"))

# Preparar audio config
AUDIO_CFG = CONFIG.get("audio", {})
RECORD_SECONDS = int(AUDIO_CFG.get("record_seconds", 5))
SAMPLERATE = int(AUDIO_CFG.get("samplerate", 16000))
CHANNELS = int(AUDIO_CFG.get("channels", 1))

# Playback commands (no usados pero los dejamos)
PLAY_CMD = CONFIG.get("playback", {}).get("video_player_cmd", "mpv --no-audio-display --fs --really-quiet {file}")
KILL_CMD = CONFIG.get("playback", {}).get("video_player_kill", "pkill -f \"mpv --no-audio-display --fs --really-quiet\"")

# API keys (tolerant to different key names in config.json)
APIS = CONFIG.get("apis", {}) or {}
# common variants mapping
GEMINI_KEY = APIS.get("gemini_api_key") or APIS.get("GEMINI_KEY") or APIS.get("gemini_key")
ELEVEN_KEY = APIS.get("eleven_api_key") or APIS.get("ELEVEN_KEY") or APIS.get("eleven_key")
# accept d_id_api_key or D_ID_KEY etc.
D_ID_KEY = APIS.get("d_id_api_key") or APIS.get("D_ID_KEY") or APIS.get("d_id_key") or APIS.get("d_id_api")
OPENAI_KEY = APIS.get("openai_api_key") or APIS.get("OPENAI_KEY") or APIS.get("openai_key")

# Firebase init (perezoso) -- usar safe_path y aceptar bucket opcional
FIREBASE_CONF = CONFIG.get("firebase", {})
FIREBASE_AVAILABLE = False
try:
	import firebase_admin
	from firebase_admin import credentials, firestore, storage  # storage importado aquí

	cred_path = FIREBASE_CONF.get("service_account")
	bucket_name = FIREBASE_CONF.get("storage_bucket")

	if cred_path:
		sp = safe_path(cred_path)
		if not Path(sp).exists():
			dbg(f"Firebase service account no encontrado en: {sp}")
			FIREBASE_AVAILABLE = False
		else:
			cred = credentials.Certificate(sp)
			# inicializar app con opción de bucket si está definido
			app_opts = {}
			if bucket_name:
				app_opts["storageBucket"] = bucket_name
			try:
				firebase_admin.initialize_app(cred, app_opts or None)
				FIREBASE_DB = firestore.client()
				# probar conexión a storage
				try:
					if bucket_name:
						storage.bucket(bucket_name)
					else:
						storage.bucket()
					FIREBASE_AVAILABLE = True
					dbg("Firebase (Firestore y Storage) inicializado OK.")
				except Exception as e:
					dbg("Firebase storage init error: " + repr(e))
					FIREBASE_AVAILABLE = False
			except Exception as e:
				dbg("Firebase initialize_app fallo: " + repr(e))
				FIREBASE_AVAILABLE = False
	else:
		dbg("Firebase no inicializado: 'service_account' falta en config.json")
		FIREBASE_AVAILABLE = False

except Exception as e:
	dbg("Firebase no inicializado (Exception): " + repr(e))
	FIREBASE_AVAILABLE = False

# NFC setup
NFC_AVAILABLE = False
try:
    from mfrc522 import SimpleMFRC522
    NFC_AVAILABLE = True
except Exception:
    try:
        import MFRC522
        NFC_AVAILABLE = True
    except Exception:
        NFC_AVAILABLE = False
        dbg("Lector NFC no disponible o librería faltante.")

# Cola y estado global
event_q = queue.Queue(maxsize=16)
STATE = {
    "mode": "oobe",
    "last_activity": time.time(),
    "current_video_proc": None,
    "stop_flag": False,
    "last_played_temp": None
}

# sounddevice (opcional)
SD_AVAILABLE = False
try:
    import sounddevice as sd
    import wave
    SD_AVAILABLE = True
except Exception as e:
    dbg("sounddevice no disponible: " + repr(e))
    auto_install = CONFIG.get("auto_install_deps", False)
    err_txt = repr(e).lower()
    needs_numpy = ("numpy" in err_txt) or ("num" in err_txt and "must be installed" in err_txt)
    if needs_numpy and auto_install:
        event_q.put(("log", "Falta NumPy: intentando instalar dependencias automáticamente..."))
        ok1 = try_install("numpy")
        ok2 = try_install("sounddevice")
        if ok1 or ok2:
            try:
                importlib.invalidate_caches()
                import numpy  # type: ignore
                import sounddevice as sd  # type: ignore
                import wave
                SD_AVAILABLE = True
                event_q.put(("log", "Dependencias instaladas: sounddevice disponible"))
                dbg("Dependencias instaladas, SD_AVAILABLE=True")
            except Exception as e2:
                dbg("Re-import fallo tras instalación: " + repr(e2))
                event_q.put(("log", "No se pudo activar sounddevice tras instalación automática. Revisa logs."))
    else:
        event_q.put(("log", "sounddevice no disponible; grabación local no funcionará"))

import requests

# pygame opcional
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False
    dbg("pygame no disponible o fallo inicializando mixer.")

# play mpv embedded
def play_video_embedded(widget, filepath, loop=False):
    # start a new mpv instance without killing previous (now managed separately)
    if not Path(filepath).exists():
        dbg("Video no encontrado: " + filepath)
        return None
    try:
        widget.update_idletasks()
        wid = widget.winfo_id()
        try:
            w = widget.winfo_width()
            h = widget.winfo_height()
            if w <= 1 or h <= 1:
                w, h = 640, 480
        except Exception:
            w, h = 640, 480
        cmd = [
            "mpv",
            "--no-border",
            "--hwdec=no",
            "--force-window=no",
            "--wid=" + str(wid),
            f"--autofit={w}x{h}",
            "--mute=no",
            "--fullscreen=no",
            "--loop-file=yes" if loop else "--loop-file=no",
            filepath
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc
    except Exception as e:
        dbg("play_video_embedded ex: " + repr(e))
        return None

def stop_embedded_video():
    try:
        idle_proc = STATE.get("idle_proc")
        if idle_proc and idle_proc.poll() is None:
            idle_proc.terminate()
            try:
                idle_proc.wait(timeout=1.0)
            except Exception:
                idle_proc.kill()
    except Exception:
        pass
    try:
        gen_proc = STATE.get("generated_proc")
        if gen_proc and gen_proc.poll() is None:
            gen_proc.terminate()
            try:
                gen_proc.wait(timeout=1.0)
            except Exception:
                gen_proc.kill()
    except Exception:
        pass
    STATE["idle_proc"] = None
    STATE["generated_proc"] = None
    STATE.pop("idle_path", None)
    STATE.pop("generated_path", None)
    STATE.pop("generated_temp", None)

def record_audio_wav(seconds=RECORD_SECONDS, samplerate=SAMPLERATE, channels=CHANNELS):
    event_q.put(("status", {"state": "recording", "msg": "Grabando audio..."}))
    global SD_AVAILABLE
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpname = tmp.name
    tmp.close()
    dbg(f"Intentando grabar audio {seconds}s -> {tmpname}")
    # Reset stop flag if present (caller _start_recording sets it too)
    try:
        STATE["stop_recording"] = False
    except Exception:
        pass

    # Prefer sounddevice: read in small chunks and stop when user releases (STATE["stop_recording"]=True)
    if SD_AVAILABLE:
        try:
            import wave as _wave
            frames = []
            chunk_secs = 0.1
            chunk_frames = int(samplerate * chunk_secs)
            with sd.RawInputStream(samplerate=samplerate, channels=channels, dtype='int16') as stream:
                start = time.time()
                while not STATE.get("stop_recording") and (time.time() - start) < seconds:
                    try:
                        data = stream.read(chunk_frames)[0]  # raw bytes
                        frames.append(data)
                    except Exception:
                        # transient read error, continue
                        time.sleep(0.01)
                # if user never released, we may have reached timeout
            # write frames to wav
            try:
                wf = _wave.open(tmpname, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                for chunk in frames:
                    wf.writeframes(chunk)
                wf.close()
            except Exception as e:
                dbg("Error escribiendo WAV (sounddevice path): " + repr(e))
                if os.path.exists(tmpname):
                    os.unlink(tmpname)
                raise
            event_q.put(("log", f"Grabación completada (sounddevice): {tmpname}"))
            return tmpname
        except Exception as e:
            dbg("Error grabando audio (sounddevice path): " + repr(e))
            SD_AVAILABLE = False
            # caer al fallback de arecord/ffmpeg

    arecord_path = shutil.which("arecord")
    if arecord_path:
        try:
            # start arecord without explicit -d so we can terminate it when user releases
            cmd = [arecord_path, "-f", "S16_LE", "-r", str(samplerate), "-c", str(channels), "-t", "wav", tmpname]
            dbg("Grabando con arecord (duración indefinida): " + " ".join(shlex.quote(p) for p in cmd))
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            start = time.time()
            # Wait until user releases or process ends or a safety timeout
            safety_timeout = seconds + 10
            while proc.poll() is None and not STATE.get("stop_recording") and (time.time() - start) < safety_timeout:
                time.sleep(0.1)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()
            # verify file exists and has more than WAV header
            if os.path.exists(tmpname) and os.path.getsize(tmpname) > 44:
                event_q.put(("log", f"Grabación completada (arecord): {tmpname}"))
                return tmpname
            else:
                raise RuntimeError("arecord produced empty or invalid file")
        except Exception as e:
            event_q.put(("log", f"arecord error: {repr(e)}"))
            if os.path.exists(tmpname):
                try:
                    os.unlink(tmpname)
                except Exception:
                    pass
            # fall through to ffmpeg fallback

    ffmpeg_path = shutil.which("ffmpeg") or shutil.which("avconv")
    if ffmpeg_path:
        try:
            cmd = [ffmpeg_path, "-y", "-f", "alsa", "-ac", str(channels), "-ar", str(samplerate), "-t", str(seconds), "-i", "default", tmpname]
            dbg("Grabando con ffmpeg: " + " ".join(shlex.quote(p) for p in cmd))
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=seconds + 15)
            event_q.put(("log", f"Grabación completada (ffmpeg): {tmpname}"))
            return tmpname
        except Exception as e:
            event_q.put(("log", f"ffmpeg error: {repr(e)}"))
            if os.path.exists(tmpname):
                try:
                    os.unlink(tmpname)
                except Exception:
                    pass
            raise

    event_q.put(("log", "No hay backend de grabación disponible (sounddevice / arecord / ffmpeg)"))
    if os.path.exists(tmpname):
        try:
            os.unlink(tmpname)
        except Exception:
            pass
    raise RuntimeError("No available audio recorder (sounddevice/arecord/ffmpeg)")
# Gemini/OpenAI helpers
try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except Exception:
    GEMINI_SDK_AVAILABLE = False

def openai_transcribe(wav_path):
    if not OPENAI_KEY:
        raise RuntimeError("OpenAI API key not configured (openai_api_key in config.json)")
    try:
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
        with open(wav_path, "rb") as f:
            files = {"file": f}
            data = {"model": "whisper-1"}
            dbg("Enviando a OpenAI Whisper HTTP")
            r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        if r.status_code in (200, 201):
            try:
                return r.json().get("text") or r.text
            except Exception:
                return r.text
        else:
            dbg("OpenAI Whisper failed status=" + str(r.status_code) + ": " + r.text[:400])
            raise RuntimeError(f"Transcripción fallida status={r.status_code}")
    except Exception as e:
        dbg("openai_transcribe error: " + repr(e))
        raise

def ensure_gemini_client():
    if not GEMINI_SDK_AVAILABLE:
        raise RuntimeError("google-genai no está instalado. Ejecuta: pip install -U google-genai")
    api_key = os.environ.get("GEMINI_API_KEY") or CONFIG.get("apis", {}).get("gemini_api_key")
    if not api_key:
        raise RuntimeError("No se encontró GEMINI_API_KEY en environment ni en config.json")
    client = genai.Client(api_key=api_key)
    return client

def gemini_chat(prompt_text):
    try:
        client = ensure_gemini_client()
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt_text)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        dbg("gemini_chat error: " + repr(e))
        raise

def gemini_transcribe(wav_path):
    if not GEMINI_KEY:
        raise RuntimeError("Gemini API key no configurada (gemini_api_key en config.json)")
    last_exc = None
    try:
        audio_type = getattr(types, "Audio", None)
    except Exception:
        audio_type = None
    if GEMINI_SDK_AVAILABLE and audio_type:
        try:
            client = ensure_gemini_client()
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
            audio_obj = audio_type(data=audio_bytes, encoding="wav")
            resp = client.models.generate_content(model="gemini-1.5-pro", contents=[audio_obj])
            return getattr(resp, "text", str(resp))
        except Exception as e:
            dbg("gemini_transcribe (genai) error: " + repr(e))
            last_exc = e
    # Google Speech-to-Text path
    try:
        sa_path = FIREBASE_CONF.get("service_account")
        from google.cloud import speech_v1 as speech
        client = None
        if sa_path:
            try:
                client = speech.SpeechClient.from_service_account_file(safe_path(sa_path))
                dbg("Google STT: usando service account from config")
            except Exception as e:
                dbg("Google STT: fallo al usar service account: " + repr(e))
                client = None
        if client is None:
            try:
                client = speech.SpeechClient()
                dbg("Google STT: SpeechClient() creado sin service account explicito")
            except Exception as e:
                dbg("Google STT: no se pudo crear SpeechClient: " + repr(e))
                client = None
        if client:
            with open(wav_path, "rb") as f:
                content = f.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLERATE,
                language_code=CONFIG.get("audio", {}).get("language") or "es-ES",
                audio_channel_count=CHANNELS
            )
            dbg("Google STT: enviando request")
            resp = client.recognize(config=config, audio=audio)
            parts = []
            for r in resp.results:
                if r.alternatives:
                    parts.append(r.alternatives[0].transcript)
            result = " ".join(parts).strip()
            if result:
                return result
            last_exc = RuntimeError("Google STT no devolvió transcripción")
    except Exception as e:
        dbg("Google STT error: " + repr(e))
        last_exc = last_exc or e
    if last_exc:
        raise last_exc
    raise RuntimeError("No se pudo transcribir el audio: falta soporte de STT configurado")

# ElevenLabs TTS
def eleven_tts(text, out_wav_path):
    if not ELEVEN_KEY:
        raise RuntimeError("ElevenLabs API key no configurada")
    voice_id = PERSON.get("eleven_voice_id")
    if not voice_id:
        raise RuntimeError("Voice ID de ElevenLabs no configurado para la persona")

    # Intentar usar SDK ElevenLabs (preferido) con modelo eleven_v3
    try:
        from elevenlabs.client import ElevenLabs  # type: ignore
        client = ElevenLabs(api_key=ELEVEN_KEY)
        # model_id = "eleven_v3" (alpha expressive model)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_v3",
            output_format="mp3_44100_128",
        )
        # audio puede ser bytes o file-like
        if isinstance(audio, (bytes, bytearray)):
            with open(out_wav_path, "wb") as f:
                f.write(audio)
            return out_wav_path
        elif hasattr(audio, "read"):
            with open(out_wav_path, "wb") as f:
                f.write(audio.read())
            return out_wav_path
        else:
            dbg("ElevenLabs SDK devolvió tipo inesperado: " + repr(type(audio)))
            # caer al fallback HTTP abajo
    except Exception as e:
        dbg("ElevenLabs SDK no disponible o fallo: " + repr(e))

    # Fallback HTTP (compatible con implementaciones previas)
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"}
        payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}, "model_id": "eleven_v3"}
        r = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        if r.status_code in (200, 201):
            with open(out_wav_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return out_wav_path
        else:
            dbg("ElevenLabs HTTP TTS fallo: " + r.text[:400])
            raise RuntimeError("ElevenLabs TTS falló (HTTP)")
    except Exception as e:
        dbg("ElevenLabs ex (fallback): " + repr(e))
        raise

def download_to_temp(url, suffix=".mp4"):
    """
    Descarga URL a archivo temporal. Verifica Content-Type para evitar guardar HTML.
    Lanza RuntimeError si el content-type no es compatible con el suffix pedido (imagen/video/audio).
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmpname = tmp.name
    tmp.close()
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").lower()
            # validar que no sea HTML (landing pages como ibb.co devuelven text/html)
            if ctype.startswith("text/") or "html" in ctype:
                # sugerir usar la URL directa a la imagen si detectamos HTML
                raise RuntimeError(f"URL no parece ser media (Content-Type={ctype}). Si es una imagen de ibb.co usa la URL directa (i.ibb.co/...).")
            # opcional: si se pidió suffix de imagen y el content-type no es image, advertir
            if suffix and suffix.lower().endswith((".jpg", ".jpeg", ".png")) and not ctype.startswith("image/"):
                raise RuntimeError(f"Se esperaba imagen pero Content-Type={ctype}")
            # escribir stream en archivo
            with open(tmpname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return tmpname
    except Exception as e:
        dbg("Descarga fallo: " + repr(e))
        if os.path.exists(tmpname):
            try:
                os.unlink(tmpname)
            except Exception:
                pass
        raise

# --- NEW: helper para cargar contexto de la persona ---
def load_person_context(person):
    """
    Si person contiene 'context_prompt_file' y existe, devuelve su contenido (string).
    Devuelve cadena vacía si no hay archivo o falló la lectura.
    """
    try:
        path = person.get("context_prompt_file")
        if not path:
            return ""
        # safe_path ya normaliza rutas absolut/relativas y deja URLs tal cual
        path = safe_path(path)
        if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
            # no esperamos contextos remotos normalmente; evitar intentos automáticos
            dbg("Context prompt es URL, no se descargará automáticamente: " + path)
            return ""
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                return fh.read()
    except Exception as e:
        dbg("load_person_context fallo: " + repr(e))
    return ""

def upload_to_firebase_storage(local_path, max_tries=3):
    """
    Sube un fichero a Firebase Storage y devuelve la URL pública.
    Lanza RuntimeError si falla.
    """
    if not FIREBASE_AVAILABLE:
        raise RuntimeError("Firebase no está disponible para subir el audio.")

    try:
        # Obtener el nombre del bucket desde la config
        bucket_name = FIREBASE_CONF.get("storage_bucket")

        # import local dentro de la función según lo solicitado
        from firebase_admin import storage
        import uuid
        import datetime

        # elegir bucket por defecto si no se especifica nombre explícito
        if bucket_name:
            bucket = storage.bucket(bucket_name)
        else:
            bucket = storage.bucket()

        # Generar un nombre de archivo único para evitar colisiones
        fname = f"audio_uploads/{uuid.uuid4()}_{Path(local_path).name}"
        blob = bucket.blob(fname)

        # Subir el archivo
        blob.upload_from_filename(local_path)

        # Intentar marcar público; si no es posible, generar signed url temporal
        public_url = None
        try:
            blob.make_public()
            public_url = getattr(blob, "public_url", None)
        except Exception:
            dbg("Advertencia: no se pudo marcar como público con blob.make_public(); intentando signed URL")
        if not public_url:
            try:
                public_url = blob.generate_signed_url(expiration=datetime.timedelta(hours=1))
            except Exception as e:
                dbg("No se pudo generar signed_url: " + repr(e))
                # fallback al public_url si existe
                public_url = getattr(blob, "public_url", None)

        if not public_url:
            raise RuntimeError("No se obtuvo URL pública del objeto subido a Firebase Storage.")

        dbg(f"Audio subido a Firebase Storage: {public_url}")
        return public_url

    except Exception as e:
        dbg("Firebase Storage upload ex: " + repr(e))
        raise RuntimeError(f"No se pudo subir audio a Firebase Storage: {repr(e)}")

# D-ID create talking head (modernized with polling)
def d_id_generate_video(photo_path, text_script, audio_path=None):
    """
    Crea un 'talk' en D-ID usando la API JSON y devuelve la URL final del video.
    Si audio_path se provee, se sube el audio a transfer.sh y se envía script type=audio con audio_url.
    """
    if not D_ID_KEY and not CONFIG.get("d_id_api_key"):
        raise RuntimeError("D-ID API key no configurada (config.apis.D_ID_KEY o config.d_id_api_key)")

    import base64

    cfg_auth_type = (CONFIG.get("d_id_auth_type") or "").strip().lower()
    cfg_api_key = CONFIG.get("d_id_api_key") or None
    effective_key = cfg_api_key if cfg_api_key else D_ID_KEY

    def mask_key(k):
        try:
            if not k:
                return "<empty>"
            s = str(k)
            if len(s) <= 12:
                return s[:4] + "..." + s[-3:]
            return s[:6] + "..." + s[-6:]
        except Exception:
            return "<key?>"

    api_url = "https://api.d-id.com/talks"
    base_headers = {"accept": "application/json", "content-type": "application/json"}

    # Build payload depending on whether we have local audio
    payload = {}
    use_external_tts = True

    if audio_path:
        # Use our pre-generated audio: upload to Firebase Storage and send audio_url to D-ID
        dbg("D-ID: audio_path provided, subiendo audio a Firebase Storage para D-ID")
        audio_url = upload_to_firebase_storage(audio_path)
        dbg("D-ID: uploaded audio_url=" + audio_url)
        payload["script"] = {"type": "audio", "audio_url": audio_url}
        use_external_tts = False
    else:
        # text script: force provider from PERSON / config (no guessing)
        if ELEVEN_KEY:
            provider = {"type": "elevenlabs"}
            vid = PERSON.get("eleven_voice_id")
            if vid:
                provider["voice_id"] = vid
        else:
            provider = {"type": "microsoft"}
        payload["script"] = {"type": "text", "provider": provider, "input": text_script}
        use_external_tts = True

    # source image handling (URL or local -> data URI)
    if isinstance(photo_path, str) and (photo_path.startswith("http://") or photo_path.startswith("https://")):
        payload["source_url"] = photo_path
    else:
        if not Path(photo_path).exists():
            raise RuntimeError("Foto local para D-ID no encontrada: " + str(photo_path))
        try:
            with open(photo_path, "rb") as fh:
                b = fh.read()
            b64 = base64.b64encode(b).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{b64}"
            payload["source_image"] = data_uri
        except Exception as e:
            dbg("D-ID local image encode error: " + repr(e))
            raise

    # optional: driver/config
    driver = CONFIG.get("d_id_driver") or PERSON.get("d_id_driver")
    if driver:
        payload["driver_url"] = driver
    cfg = {}
    # mantener el aspecto original (no crop) solicitando stitch por defecto; puede anularse con config.d_id_stitch=false
    if CONFIG.get("d_id_stitch", True):
        cfg["stitch"] = True
    if CONFIG.get("d_id_persist", False):
        cfg["persist"] = True
    user_data = CONFIG.get("d_id_user_data")
    if user_data:
        cfg["user_data"] = str(user_data)[:1000]
    if cfg:
        payload["config"] = cfg

    # helper to post
    def try_post(headers):
        try:
            rr = requests.post(api_url, headers=headers, json=payload, timeout=30)
            return rr.status_code, rr
        except Exception as e:
            dbg("D-ID POST request exception: " + repr(e))
            return None, e

    # build auth variants (respect explicit config)
    auth_variants = []
    if cfg_auth_type:
        ak = effective_key or ""
        t = cfg_auth_type
        if t == "bearer":
            auth_variants.append(("Authorization", f"Bearer {ak}"))
        elif t == "basic":
            try:
                if isinstance(ak, str) and ":" in ak:
                    b64 = base64.b64encode(ak.encode()).decode()
                else:
                    b64 = base64.b64encode(f"api:{ak}".encode()).decode()
                auth_variants.append(("Authorization", f"Basic {b64}"))
            except Exception:
                auth_variants.append(("Authorization", f"Basic {ak}"))
        elif t in ("x-api-key", "x_api_key"):
            auth_variants.append(("x-api-key", ak))
        elif t in ("api-key", "api_key"):
            auth_variants.append(("Api-Key", ak))
        elif t in ("none", "noauth"):
            auth_variants = [None]
        else:
            auth_variants.append(("Authorization", f"Bearer {ak}"))
    else:
        if isinstance(effective_key, str) and effective_key.strip().lower().startswith("bearer "):
            auth_variants.append(("Authorization", effective_key.strip()))
        else:
            try:
                if isinstance(effective_key, str) and ":" in effective_key:
                    possible_token = effective_key.split(":")[-1].strip()
                    if possible_token:
                        auth_variants.append(("Authorization", f"Bearer {possible_token}"))
            except Exception:
                pass
            if effective_key:
                auth_variants.append(("Authorization", f"Bearer {effective_key}"))
            try:
                if isinstance(effective_key, str) and ":" in effective_key:
                    b64 = base64.b64encode(effective_key.encode()).decode()
                    auth_variants.append(("Authorization", f"Basic {b64}"))
            except Exception:
                pass
            if effective_key:
                auth_variants.append(("x-api-key", effective_key))
                auth_variants.append(("Api-Key", effective_key))
                auth_variants.append(("Authorization", effective_key))

    last_err = None

    # Try variants
    for vh in auth_variants:
        try:
            if not vh:
                continue
            header_name, header_value = vh
        except Exception:
            continue

        headers = dict(base_headers)
        headers[header_name] = header_value
        # only add external TTS header when we are NOT supplying audio directly
        if use_external_tts and ELEVEN_KEY:
            try:
                headers["x-api-key-external"] = json.dumps({"elevenlabs": ELEVEN_KEY})
            except Exception:
                pass

        dbg(f"D-ID trying POST with header {header_name}={mask_key(header_value)}")
        status, resp = try_post(headers)
        if status is None:
            last_err = resp
            dbg(f"D-ID POST attempt exception with header {header_name}: {repr(resp)}")
            continue

        dbg(f"D-ID POST status={status} (header {header_name} used)")
        if status == 401:
            try:
                dbg("D-ID POST returned 401 body: " + (resp.text[:800] if hasattr(resp, "text") else repr(resp)))
            except Exception:
                pass
            last_err = RuntimeError("Unauthorized with header " + header_name)
            continue

        if status not in (200, 201):
            try:
                dbg(f"D-ID POST failed body: {resp.text[:1500] if hasattr(resp, 'text') else repr(resp)}")
            except Exception:
                pass
            last_err = RuntimeError(f"D-ID POST failed status={status}")
            continue

        # success: parse json
        try:
            job = resp.json()
        except Exception:
            dbg("D-ID POST success but response not JSON: " + (resp.text[:800] if hasattr(resp, "text") else "<no text>"))
            last_err = RuntimeError("D-ID POST returned non-JSON response")
            continue

        # immediate result url?
        for key in ("result_url", "video_url", "output_url"):
            if key in job and job[key]:
                return job[key]

        job_id = job.get("id") or job.get("talk_id")
        if not job_id:
            dbg("D-ID POST did not return job id; response keys: " + ", ".join(job.keys()))
            last_err = RuntimeError("D-ID POST no devolvió id de talk ni result_url")
            continue

        poll_url = f"{api_url}/{job_id}"
        dbg(f"D-ID: polling {poll_url} (using header {header_name})")
        for _ in range(90):
            try:
                rr = requests.get(poll_url, headers=headers, timeout=30)
                if rr.status_code == 200:
                    j2 = rr.json()
                    status2 = j2.get("status")
                    candidates = []
                    candidates.extend([j2.get("result_url"), j2.get("video_url"), j2.get("output_url")])
                    if isinstance(j2.get("files"), list):
                        for f in j2.get("files"):
                            if isinstance(f, dict) and f.get("url"):
                                candidates.append(f.get("url"))
                    if isinstance(j2.get("results"), dict):
                        candidates.append(j2["results"].get("result_url") or j2["results"].get("video_url"))
                    for c in candidates:
                        if c:
                            dbg(f"D-ID: encontrado resultado en polling: {c}")
                            return c
                    if status2 in ("error", "rejected"):
                        dbg("D-ID job failed status: " + repr(j2))
                        raise RuntimeError("D-ID job failed: " + repr(j2))
                else:
                    dbg(f"D-ID polling HTTP status={rr.status_code} body={rr.text[:200]}")
            except Exception as e:
                dbg("D-ID poll ex: " + repr(e))
            time.sleep(2)
        raise RuntimeError("D-ID polling timeout or no result_url produced")

    guidance = ("D-ID authentication failed. Check config.json:\n"
                 " - Set config.apis.D_ID_KEY to your API key string (or use top-level config.d_id_api_key)\n"
                 " - If D-ID expects a specific header, set config.d_id_auth_type to one of: bearer, basic, x-api-key, api-key, none\n"
                 "Example:\n"
                 '  \"d_id_auth_type\": \"bearer\",\n'
                 '  \"d_id_api_key\": \"YOUR_TOKEN_HERE\"\n'
                 "If you are unsure, get the 'Bearer <token>' value from D-ID dashboard and put it as d_id_api_key.")
    dbg(guidance)
    if last_err:
        raise last_err
    raise RuntimeError("D-ID POST failed with unknown error. " + guidance)
# NFC thread (clean polling implementation)
def nfc_thread_func():
    if not NFC_AVAILABLE:
        dbg("NFC thread abortado: biblioteca no disponible.")
        return
    try:
        try:
            from mfrc522 import SimpleMFRC522
            reader = SimpleMFRC522()
            dbg("NFC: usando SimpleMFRC522 (modo simple)")
            # SimpleMFRC522 has a blocking read method; we poll gently to remain responsive
            while not STATE["stop_flag"]:
                try:
                    id_text = None
                    # SimpleMFRC522.read() is blocking; avoid calling it in this loop
                    time.sleep(1)
                except Exception:
                    time.sleep(1)
        except Exception:
            # fallback to low-level MFRC522 polling implementation
            import MFRC522
            MIFARE = MFRC522.MFRC522()
            dbg("NFC: usando MFRC522.MFRC522 (polling)")
            while not STATE["stop_flag"]:
                try:
                    (status, TagType) = MIFARE.MFRC522_Request(MIFARE.PICC_REQIDL)
                    if status == MIFARE.MI_OK:
                        (status2, uid) = MIFARE.MFRC522_Anticoll()
                        if status2 == MIFARE.MI_OK:
                            uid_str = ''.join([format(x, '02x') for x in uid])
                            dbg("NFC tag detectado: " + uid_str)
                            mapped = CONFIG.get("nfc_map", {}).get(uid_str)
                            if mapped:
                                STATE["last_activity"] = time.time()
                                video_path = safe_path(mapped)
                                if Path(video_path).exists():
                                    event_q.put(("play_generated_video", {"path": video_path, "temp": False}))
                                else:
                                    if str(mapped).startswith("http"):
                                        try:
                                            dbg(f"NFC: Descargando video {mapped}")
                                            tmpv = download_to_temp(mapped, suffix=".mp4")
                                            event_q.put(("play_generated_video", {"path": tmpv, "temp": True}))
                                        except Exception as e:
                                            dbg(f"NFC: Falla al descargar video: {e}")
                                            plan_b_play()
                    time.sleep(0.6)
                except Exception as e:
                    dbg("NFC loop ex: " + repr(e))
                    time.sleep(1)
    except Exception as e:
        dbg("NFC hilo fallo: " + repr(e))

# Firebase poller
def firebase_poller(root_widget, interval=10):
    if not FIREBASE_AVAILABLE:
        dbg("Firebase poller no disponible.")
        return
    doc_path = FIREBASE_CONF.get("device_status_doc")
    if not doc_path:
        dbg("Firebase: device_status_doc no configurado.")
        return
    parts = doc_path.split("/")
    if len(parts) < 2:
        dbg("device_status_doc formato invalido.")
        return
    while not STATE["stop_flag"]:
        try:
            doc_ref = FIREBASE_DB.document(doc_path)
            snap = doc_ref.get()
            if snap.exists:
                data = snap.to_dict()
                event_q.put(("firebase_status", data))
        except Exception as e:
            dbg("Firebase poll ex: " + repr(e))
        time.sleep(interval)

# UI class (clean)
import tkinter as tk
from PIL import Image, ImageTk

class CajaMemoraUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Caja Memora Demo")
        # para debug: desactivar fullscreen por defecto (cambiar a True en la Pi cuando confirmes)
        try:
            self.root.attributes("-fullscreen", False)
        except Exception:
            pass
        # tamaño de ventana por defecto (útil en desktop)
        try:
            self.root.geometry("1280x720")
        except Exception:
            pass
        self.root.configure(background="#111111")
        self.root.update_idletasks()
        try:
            win_w = self.root.winfo_width()
            win_h = self.root.winfo_height()
            # si devuelve 1 (a veces al inicio), fallback a 1280x720
            if win_w <= 1 or win_h <= 1:
                win_w, win_h = 1280, 720
        except Exception:
            win_w, win_h = 1280, 720
        self.width = win_w
        self.height = win_h

        # FRAME principal donde mpv embebe su ventana
        self.video_frame = tk.Frame(root, bg="black", width=self.width, height=self.height)
        self.video_frame.pack(fill="both", expand=True)
        self.video_frame.update_idletasks()

        # Idle video frame (background, always looping)
        self.idle_video_frame = tk.Frame(self.video_frame, bg="black")
        self.idle_video_frame.place(x=0, y=0, relwidth=1, relheight=1)

        # Generated video frame (on top, raised when playing generated video)
        self.generated_video_frame = tk.Frame(self.video_frame, bg="black")
        self.generated_video_frame.place(x=0, y=0, relwidth=1, relheight=1)
        self.generated_video_frame.lower()  # Start lowered so idle is visible

        # overlay para mostrar avatar / slides / texto
        # crear Canvas sin pasar bg="" (algunos sistemas/tk no aceptan cadena vacía como color)
        # Si se necesita "transparencia" real, hay que usar otras técnicas; aquí evitamos el crash.
        self.overlay_canvas = tk.Canvas(self.video_frame, highlightthickness=0, width=self.width, height=self.height)
        self.overlay_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # Panel de estado en esquina superior izquierda (usar rel coords para ser robusto)
        status_frame = tk.Frame(self.root, bg="#111111")
        status_frame.place(relx=0.01, rely=0.01, anchor="nw")

        self.status_label = tk.Label(status_frame, text="estado: oobe — iniciando", bg="#111111", fg="white")
        self.status_label.pack(side="left", padx=4)

        # simple status boxes (keys used by set_status)
        self.status_boxes = {}
        for key in ("oobe", "idle", "playing", "standby", "listening", "transcribing", "tts", "d-id", "recording"):
            lbl = tk.Label(status_frame, text=key, bg="#222222", fg="#aaaaaa", padx=6, pady=2)
            lbl.pack(side="left", padx=2)
            self.status_boxes[key] = lbl

        # log text (readonly) — ubicar en parte inferior con relcoords
        self.log_text = tk.Text(self.root, height=6, bg="#111111", fg="white", state="disabled")
        self.log_text.place(relx=0.01, rely=0.74, relwidth=0.65, relheight=0.22)

        # hablar button — ahora bind para press-and-hold (no command)
        self.hablar_btn = tk.Button(self.root, text="HABLAR", bg="#2b8cff", fg="white")
        self.hablar_btn.place(relx=0.87, rely=0.82, relwidth=0.12, relheight=0.12)
        # bind press/release
        try:
            self.hablar_btn.bind("<ButtonPress-1>", self._on_hablar_press)
            self.hablar_btn.bind("<ButtonRelease-1>", self._on_hablar_release)
            # touch cancel (leave) as stop too
            self.hablar_btn.bind("<Leave>", self._on_hablar_release)
        except Exception:
            # fallback to command if bind fails
            self.hablar_btn.config(command=self.on_hablar_pressed)
        # track press state
        self._hablar_pressed = False

        # otros flags e imagenes
        self.use_ttk = False
        self.avatar_img = None
        try:
            photo = PERSON.get("photo")
            img = None
            if photo:
                # Si es URL, intentar descargarla temporalmente
                if isinstance(photo, str) and (photo.startswith("http://") or photo.startswith("https://")):
                    try:
                        tmpimg = download_to_temp(photo, suffix=".jpg")
                        img = Image.open(tmpimg).convert("RGB")
                    except Exception as e:
                        dbg("Avatar remote download failed: " + repr(e))
                        img = None
                else:
                    # local path
                    try:
                        if Path(photo).exists():
                            img = Image.open(photo).convert("RGB")
                    except Exception:
                        img = None
            if img:
                img.thumbnail((int(self.width * 0.3), int(self.height * 0.45)), Image.Resampling.LANCZOS)
                self.avatar_img = ImageTk.PhotoImage(img)
            else:
                self.avatar_img = None
        except Exception:
            self.avatar_img = None

        # slideshow
        self.slideshow_files = []
        self.slideshow_index = 0
        self.current_ss_img = None
        self.load_slideshow_images()

        # pulse state
        self._pulse_on = False

        # mostrar pantalla inicial OOBE inmediatamente
        try:
            self.show_oobe_screen()
        except Exception:
            pass

        # mantener llamada al loop de UI
        self.root.after(200, self.ui_loop)

        # asegurar que el pulso y la OOBE estén activos
        try:
            self._pulse()
        except Exception:
            pass
        try:
            self.show_oobe_screen()
        except Exception:
            pass

    def _pulse(self):
        # pulso simple para dar sensación de "activo"
        try:
            if STATE.get("mode") in ("listening","recording","playing"):
                # cambiar color del botón
                color = "#2b8cff" if self._pulse_on else "#1a6fe0"
                try:
                    if self.use_ttk:
                        self.hablar_btn.configure(style='CM.TButton')
                    else:
                        self.hablar_btn.config(bg=color)
                except Exception:
                    pass
                self._pulse_on = not self._pulse_on
            else:
                # color normal
                if not self.use_ttk:
                    self.hablar_btn.config(bg="#2b8cff")
            self.root.after(600, self._pulse)
        except Exception:
            pass

    def load_slideshow_images(self):
        ss_dir = BASE_DIR / "slideshow"
        if ss_dir.exists() and ss_dir.is_dir():
            for f in ss_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.slideshow_files.append(str(f))
        if not self.slideshow_files and PERSON.get("photo"):
            self.slideshow_files = [PERSON.get("photo")]

    def _on_hablar_press(self, ev=None):
        # On press: start recording immediately (disable button only during recording)
        try:
            self._hablar_pressed = True
            self.hablar_btn.config(relief="sunken", state="disabled", text="Grabando...")
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._start_recording, daemon=True)
            self.recording_thread.start()
        except Exception:
            pass

    def _on_hablar_release(self, ev=None):
        # On release: stop recording, enable button immediately, start processing in background
        try:
            if getattr(self, "_hablar_pressed", False):
                self._hablar_pressed = False
                self.hablar_btn.config(relief="raised", state="normal", text="HABLAR")
                # Signal stop recording and start processing
                STATE["stop_recording"] = True
                t = threading.Thread(target=handle_speech_sequence, daemon=True)
                t.start()
        except Exception:
            pass

    def _start_recording(self):
        # Handle recording in a separate thread for press-hold
        STATE["recording_in_progress"] = True
        try:
            # Record until stop signal
            wav = record_audio_wav()  # Modify record_audio_wav to respect stop signal if needed, but for now keep fixed duration
            STATE["recorded_wav"] = wav
        except Exception as e:
            event_q.put(("log", f"Recording error: {repr(e)}"))
        STATE["recording_in_progress"] = False

    def show_oobe_screen(self):
        # Start idle video loop in background during OOBE (behind overlay), like a video call
        # Ensure idle loop is running (use show_idle_video which reuses existing proc if posible)
        try:
            self.show_idle_video()
        except Exception as e:
            dbg("show_oobe_screen: no se pudo asegurar idle video: " + repr(e))
        # Keep overlay on top for QR
        try:
            self.overlay_canvas.tkraise()
        except Exception:
            pass
        self.overlay_canvas.delete("all")

        qr_path = BASE_DIR / "pairing_qr.png"
        if qr_path.exists():
            try:
                img = Image.open(qr_path)
                img.thumbnail((int(self.width*0.45), int(self.height*0.55)), Image.Resampling.LANCZOS)
                self.qr_img = ImageTk.PhotoImage(img)
                self.overlay_canvas.create_image(self.width//2, self.height//2 - 40, image=self.qr_img, anchor="center")
            except Exception:
                pass
        self.set_status("oobe", "Escanea para emparejar")

    def _finalize_generated_cleanup(self):
        """
        Final cleanup after generated video finishes + short buffer to avoid abrupt cut.
        Called via root.after to allow idle player to start underneath.
        """
        try:
            # Lower generated frame and remove temp if any
            try:
                self.generated_video_frame.lower()
            except Exception:
                pass
            tmp_path = STATE.get("generated_temp")
            if tmp_path and Path(tmp_path).exists():
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            STATE["generated_proc"] = None
            STATE["generated_path"] = None
            STATE["generated_temp"] = None
        finally:
            # clear pending flag
            STATE.pop("generated_cleanup_pending", None)
            # return to idle state
            STATE["mode"] = "idle"
            self.set_status("idle", "idle video")

    def ui_loop(self):
        # procesar cola de eventos
        while True:
            try:
                ev = event_q.get_nowait()
            except queue.Empty:
                break
            self.handle_event(ev)

        # Check idle proc
        idle_proc = STATE.get("idle_proc")
        if idle_proc and idle_proc.poll() is not None:
            # Idle proc ended, restart it
            STATE["idle_proc"] = None
            self.show_idle_video()

        # Check generated proc: if ended, don't cut immediately — ensure idle is running and finalize after small delay
        gen_proc = STATE.get("generated_proc")
        if gen_proc and gen_proc.poll() is not None:
            if not STATE.get("generated_cleanup_pending"):
                STATE["generated_cleanup_pending"] = True
                # Ensure idle is running underneath (this will reuse existing idle if present)
                try:
                    self.show_idle_video()
                except Exception:
                    pass
                # Wait a bit so mpv idle can appear smoothly, then finalize cleanup
                try:
                    self.root.after(800, lambda: self._finalize_generated_cleanup())
                except Exception:
                    # fallback immediate finalize
                    self._finalize_generated_cleanup()

        # standby por inactividad
        now = time.time()
        if now - STATE["last_activity"] > 60 and STATE["mode"] == "idle":
            STATE["mode"] = "standby"
            self.show_slideshow()

        # si estamos en idle y no hay mpv corriendo, asegurar idle loop
        if STATE.get("mode") == "idle" and not STATE.get("idle_proc"):
            # lanza idle embebido si no hay ya
            self.show_idle_video()

        # Ensure idle video plays in all modes except "playing" (generated video)
        if STATE.get("mode") != "playing" and not STATE.get("idle_proc"):
            self.show_idle_video()

        self.root.after(500, self.ui_loop)

    def set_status(self, key, msg):
        # actualizar label principal
        try:
            self.status_label.config(text=f"estado: {key} — {msg}")
        except Exception:
            pass
        # destacar el box correspondiente
        for st, lbl in self.status_boxes.items():
            if st == key:
                lbl.config(bg="#2b8cff", fg="white")
            else:
                lbl.config(bg="#222222", fg="#aaaaaa")

    def append_log(self, text):
        try:
            dbg(text)
            self.log_text.configure(state="normal")
            ts = time.strftime("%H:%M:%S")
            self.log_text.insert("end", f"[{ts}] {text}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        except Exception:
            pass

    def show_avatar(self):
        stop_embedded_video()
        try:
            self.overlay_canvas.tkraise()
        except Exception:
            pass
        self.overlay_canvas.delete("all")
        if self.avatar_img:
            self.overlay_canvas.create_image(self.width//2, self.height//2 - 20, image=self.avatar_img, anchor="center")
        else:
            self.overlay_canvas.create_text(self.width//2, self.height//2, text=PERSON.get("name"), fill="white", font=("Helvetica", 48))
        self.set_status("idle", "Avatar mostrado")
        STATE["mode"] = "idle"

    def show_idle_video(self):
        idle = PERSON.get("idle_video")
        if idle and Path(idle).exists():
            current_proc = STATE.get("idle_proc")
            # If already playing the same idle video, do nothing
            if current_proc and STATE.get("idle_path") == idle and current_proc.poll() is None:
                return
            # Start idle mpv in the idle frame
            proc = play_video_embedded(self.idle_video_frame, idle, loop=True)
            STATE["idle_proc"] = proc
            STATE["idle_path"] = idle
            # Ensure idle frame is lowered (behind generated)
            self.idle_video_frame.lower()
            # Lower overlay for video
            try:
                self.root.after(300, lambda: self.overlay_canvas.tk.call('lower', self.overlay_canvas._w))
            except Exception:
                pass
            self.set_status("idle", "idle video")
            STATE["mode"] = "idle"
        else:
            self.show_avatar()

    def show_slideshow(self):
        stop_embedded_video()
        try:
            self.overlay_canvas.tkraise()
        except Exception:
            pass
        self.overlay_canvas.delete("all")
        if not self.slideshow_files:
            self.show_avatar()
            return
        p = self.slideshow_files[self.slideshow_index % len(self.slideshow_files)]
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
            self.current_ss_img = ImageTk.PhotoImage(img)
            self.overlay_canvas.create_image(self.width//2, self.height//2 - 20, image=self.current_ss_img, anchor="center")
        except Exception:
            self.show_avatar()
        self.slideshow_index += 1
        STATE["mode"] = "standby"
        self.set_status("standby", "slideshow")
        self.root.after(6000, self.show_slideshow)

    def handle_event(self, ev):
        typ = ev[0]
        payload = ev[1] if len(ev) > 1 else None
        if typ == "status":
            self.set_status(payload.get("state"), payload.get("msg"))
        elif typ == "log":
            self.append_log(payload)
        elif typ == "enable_hablar":
            try:
                self.hablar_btn.config(state="normal", text="HABLAR")
            except Exception:
                pass
        elif typ == "show_avatar":
            try:
                self.show_avatar()
            except Exception:
                pass
        elif typ == "play_generated_video":
            try:
                if isinstance(payload, dict):
                    path = payload.get("path")
                    is_temp = bool(payload.get("temp"))
                else:
                    path = payload
                    is_temp = False
                # Ensure idle is running underneath to allow smooth transition back (no instant black)
                try:
                    self.show_idle_video()
                except Exception:
                    pass
                # Play generated video in the generated frame, raise it on top
                self.generated_video_frame.tkraise()
                proc = play_video_embedded(self.generated_video_frame, path, loop=False)
                STATE["generated_proc"] = proc
                STATE["generated_path"] = path
                STATE["generated_temp"] = path if is_temp else None
                # ensure any previous cleanup flag cleared
                STATE.pop("generated_cleanup_pending", None)
                # Lower overlay so video visible
                try:
                    self.overlay_canvas.tk.call('lower', self.overlay_canvas._w)
                except Exception:
                    pass
                self.set_status("playing", "reproduciendo generado")
                STATE["mode"] = "playing"
            except Exception as e:
                self.append_log(f"Error reproduciendo embebido: {repr(e)}")
                plan_b_play()
        elif typ == "firebase_status":
            if isinstance(payload, dict) and payload.get("mode") == "standby":
                self.show_slideshow()
            elif isinstance(payload, dict) and payload.get("mode") == "idle":
                self.show_idle_video()

# Orquestador principal
def handle_speech_sequence():
    # Wait for recording to finish if still in progress
    while STATE.get("recording_in_progress"):
        time.sleep(0.1)
    wav = STATE.pop("recorded_wav", None)
    if not wav:
        event_q.put(("status", {"state": "idle", "msg": "No audio recorded"}))
        return
    STATE["mode"] = "listening"
    STATE["last_activity"] = time.time()
    event_q.put(("status", {"state": "listening", "msg": "Procesando audio..."}))
    # Enable button immediately after recording stops (allow re-interaction)
    event_q.put(("enable_hablar", None))
    # Continue processing in background
    try:
        transcription_service = CONFIG.get("services", {}).get("transcription", "gemini")
        if transcription_service == "openai":
            transcript = openai_transcribe(wav)
        else:
            transcript = gemini_transcribe(wav)
        event_q.put(("log", f"Transcripción: {transcript}"))
    except Exception as e:
        transcript = None
        event_q.put(("log", f"STT error: {repr(e)}"))

    try:
        os.unlink(wav)
    except Exception:
        pass

    if not transcript:
        event_q.put(("status", {"state": "idle", "msg": "Transcripción fallida"}))
        plan_b_play()
        STATE["mode"] = "idle"
        try:
            event_q.put(("enable_hablar", None))
        except Exception:
            pass
        return

    # Build prompt using context if available
    context_text = load_person_context(PERSON)
    if context_text:
        full_prompt = f"{context_text}\n\nUsuario: {transcript}\n\nRespuesta:"
    else:
        full_prompt = transcript

    event_q.put(("status", {"state": "chatting", "msg": "Generando respuesta..."}))
    try:
        response_text = gemini_chat(full_prompt)
        event_q.put(("log", f"Respuesta chat: {response_text}"))
    except Exception as e:
        response_text = None
        event_q.put(("log", f"Chat error: {repr(e)}"))

    if not response_text:
        event_q.put(("status", {"state": "idle", "msg": "Chat falló"}))
        plan_b_play()
        STATE["mode"] = "idle"
        try:
            event_q.put(("enable_hablar", None))
        except Exception:
            pass
        return

    event_q.put(("status", {"state": "tts", "msg": "Generando voz..."}))
    out_audio_path = None
    try:
        out_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        out_audio_path = out_audio.name
        out_audio.close()
        eleven_tts(response_text, out_audio_path)
        event_q.put(("log", f"Audio TTS generado: {out_audio_path}"))
    except Exception as e:
        event_q.put(("log", f"TTS error: {repr(e)}"))
        out_audio_path = None

    if not out_audio_path or not Path(out_audio_path).exists():
        event_q.put(("status", {"state": "idle", "msg": "TTS falló"}))
        plan_b_play()
        STATE["mode"] = "idle"
        try:
            event_q.put(("enable_hablar", None))
        except Exception:
            pass
        return

    # Use D-ID with provided audio (d_id_generate_video will upload to Firebase)
    event_q.put(("status", {"state": "d-id", "msg": "Generando video..."}))
    try:
        video_url = d_id_generate_video(PERSON.get("photo"), response_text, audio_path=out_audio_path)
        event_q.put(("log", f"D-ID URL: {video_url}"))
    except Exception as e:
        video_url = None
        event_q.put(("log", f"D-ID error: {repr(e)}"))

    # cleanup audio file
    try:
        os.unlink(out_audio_path)
    except Exception:
        pass

    if not video_url:
        event_q.put(("status", {"state": "idle", "msg": "D-ID falló"}))
        plan_b_play()
        STATE["mode"] = "idle"
        try:
            event_q.put(("enable_hablar", None))
        except Exception:
            pass
        return

    event_q.put(("status", {"state": "downloading", "msg": "Descargando video generado..."}))
    try:
        video_path = download_to_temp(video_url, suffix=".mp4")
        event_q.put(("log", f"Video descargado: {video_path}"))
    except Exception as e:
        video_path = None
        event_q.put(("log", f"Descarga video error: {repr(e)}"))

    if not video_path:
        event_q.put(("status", {"state": "idle", "msg": "Descarga falló"}))
        plan_b_play()
        STATE["mode"] = "idle"
        try:
            event_q.put(("enable_hablar", None))
        except Exception:
            pass
        return

    event_q.put(("log", f"Reproduciendo video desde evento: {video_path}"))
    event_q.put(("play_generated_video", {"path": video_path, "temp": True}))
    event_q.put(("status", {"state": "playing", "msg": "Reproduciendo video..."}))

    STATE["mode"] = "idle"
    STATE["last_activity"] = time.time()
    try:
        event_q.put(("enable_hablar", None))
    except Exception:
        pass

def plan_b_play(person=PERSON):
    try:
        path = person.get("planb_video") or person.get("idle_video")
        if not path:
            event_q.put(("log", "plan_b_play: no hay planb configurado"))
            return False
        path = safe_path(path)
        if Path(path).exists():
            event_q.put(("log", f"plan_b_play: encolando {path}"))
            event_q.put(("play_generated_video", {"path": path, "temp": False}))
            return True
        else:
            event_q.put(("log", f"plan_b_play: archivo no encontrado {path}"))
            return False
    except Exception as e:
        event_q.put(("log", f"plan_b_play ex: {repr(e)}"))
        return False

def start_background_threads():
    if NFC_AVAILABLE:
        t_nfc = threading.Thread(target=nfc_thread_func, daemon=True)
        t_nfc.start()
    STATE["last_activity"] = time.time()

def shutdown(root):
    STATE["stop_flag"] = True
    stop_embedded_video()
    try:
        root.destroy()
    except Exception:
        pass
    sys.exit(0)

# --- ADDED: main entrypoint para inicializar UI y hilos ---
def main():
    try:
        root = tk.Tk()
    except Exception as e:
        dbg("Tkinter no disponible o fallo iniciando Tk: " + repr(e))
        raise

    try:
        ui = CajaMemoraUI(root)
    except Exception as e:
        dbg("Error inicializando UI: " + repr(e))
        try:
            root.destroy()
        except Exception:
            pass
        raise

    # iniciar hilos de background y protocolo de cierre
    start_background_threads()
    root.protocol("WM_DELETE_WINDOW", lambda: shutdown(root))

    try:
        root.mainloop()
    except KeyboardInterrupt:
        shutdown(root)
    except Exception as e:
        dbg("Error en mainloop: " + repr(e))
        try:
            shutdown(root)
        except Exception:
            pass

if __name__ == "__main__":
    main()