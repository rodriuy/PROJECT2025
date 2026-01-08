# Configuración de Credenciales

Este proyecto requiere credenciales y claves API para funcionar. Las claves están censguradas por motivos de seguridad.

## Pasos para configurar:

### 1. config.json
Copia el archivo `config.json.example` a `config.json`:
```bash
cp config.json.example config.json
```

Luego completa los valores:
- `gemini_api_key`: Tu clave de API de Google Gemini
- `eleven_api_key`: Tu clave de API de ElevenLabs
- `d_id_api_key`: Tu clave de API de D-ID
- `openai_api_key`: Tu clave de API de OpenAI
- `eleven_voice_id`: El ID de voz de ElevenLabs que deseas usar
- `d_id_person_id`: El ID de persona de D-ID
- URLs de Firebase con tu proyecto
- NFC tag ID real

### 2. serviceAccountKey.json
Descarga la clave de Firebase desde tu consola de Firebase y guárdala en `serviceAccountKey.json`:

1. Ve a Firebase Console → Tu Proyecto
2. Settings → Service Accounts
3. Generate new private key
4. Guarda el archivo descargado como `serviceAccountKey.json`

## Notas de seguridad:
- ⚠️ **NUNCA** hagas commit de `config.json` ni `serviceAccountKey.json`
- Estos archivos están incluidos en `.gitignore` para prevenir subidas accidentales
- Las credenciales son esenciales para ejecutar localmente pero deben mantenerse privadas
