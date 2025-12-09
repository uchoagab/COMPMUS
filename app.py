import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import requests
import time
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MUSIC_AI_KEY")
WORKFLOW_ID = "untitled-workflow-2708ffa"

# --- UPLOAD ---
def get_upload_url(file_path):
    """
    Tenta múltiplos serviços e ignora proxy para evitar WinError 10061.
    """
    nome_seguro = "temp_input_safe.mp3"
    try:
        shutil.copy(file_path, nome_seguro)
    except:
        nome_seguro = file_path
    
    print(f"☁️ Preparando envio de: {os.path.basename(nome_seguro)}")

    session = requests.Session()
    session.trust_env = False 

    # Tentativa 1: 0x0.st
    try:
        print("1️⃣ Tentando 0x0.st...")
        with open(nome_seguro, 'rb') as f:
            response = session.post("https://0x0.st", files={'file': f})
        if response.status_code == 200:
            link = response.text.strip()
            print(f"✅ Sucesso: {link}")
            return link
    except Exception as e:
        print(f"⚠️ 0x0 falhou: {e}")

    # Tentativa 2: Tmpfiles.org
    try:
        print("2️⃣ Tentando Tmpfiles.org...")
        with open(nome_seguro, 'rb') as f:
            response = session.post("https://tmpfiles.org/api/v1/upload", files={'file': f})
        if response.status_code == 200:
            raw_url = response.json()['data']['url']
            direct_link = raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
            print(f"✅ Sucesso: {direct_link}")
            return direct_link
    except Exception as e:
        print(f"⚠️ Tmpfiles falhou: {e}")

    raise Exception("❌ Todos os serviços de upload falharam. Verifique sua conexão.")

# --- MUSIC.AI ---
def chamar_music_ai(public_url):
    print("Iniciando Workflow na Music.ai...")
    
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "name": "Job-Textura-Final", 
        "workflow": WORKFLOW_ID,
        "params": {
            "Input1": public_url 
        }
    }
    
    response = requests.post("https://api.music.ai/api/job", json=payload, headers=headers)
    
    if response.status_code not in [200, 201]:
        raise Exception(f"Erro API Music.ai: {response.text}")
    
    job_id = response.json()['id']
    print(f"Job ID: {job_id}")
    
    # Polling
    while True:
        res = requests.get(f"https://api.music.ai/api/job/{job_id}", headers=headers)
        data = res.json()
        status = data['status']
        print(f"⏳ Status: {status}...")
        
        if status == 'SUCCEEDED':
            return data['result']
        elif status == 'FAILED':
            raise Exception("O processamento falhou lá na Music.ai.")
        
        time.sleep(3)

def baixar_stem(url, nome_salvo):
    if not url: return None
    session = requests.Session()
    session.trust_env = False
    
    res = session.get(url)
    with open(nome_salvo, 'wb') as f:
        f.write(res.content)
    return nome_salvo

# --- MAPEAMENTO ---
def buscar_link_inteligente(resultados_api, palavras_chave):
    chaves = list(resultados_api.keys())
    for chave in chaves:
        for palavra in palavras_chave:
            if palavra in chave.lower():
                print(f"Mapeado '{chave}' para '{palavra}'")
                return resultados_api[chave]
    return None

# --- PROCESSAMENTO ---
def aplicar_textura_local(stem_path, textura_path):
    if not stem_path or not textura_path: return None, 44100
    
    y_stem, sr = librosa.load(stem_path, sr=44100)
    y_textura, _ = librosa.load(textura_path, sr=44100)

    # Loop da textura
    if len(y_textura) < len(y_stem):
        reps = int(np.ceil(len(y_stem) / len(y_textura)))
        y_textura = np.tile(y_textura, reps)
    y_textura = y_textura[:len(y_stem)]

    # Envelope Follower
    frame, hop = 1024, 512
    env = librosa.feature.rms(y=y_stem, frame_length=frame, hop_length=hop)[0]
    env_interp = np.interp(np.arange(len(y_stem)), np.arange(len(env)) * hop, env)
    env_interp = env_interp / (np.max(env_interp) + 1e-9)
    
    return y_textura * env_interp, sr

# --- PRINCIPAL ---
def processar_tudo(input_music, tex_vocal, tex_drums, tex_bass, tex_guitar, tex_other):
    try:
        print("\n--- 1. UPLOAD ---")
        url = get_upload_url(input_music)
        
        print("\n--- 2. PROCESSAMENTO ---")
        res_api = chamar_music_ai(url)
        
        print(f"Chaves recebidas: {list(res_api.keys())}")

        urls_stems = {
            "Vocals": buscar_link_inteligente(res_api, ["vocal", "voice", "voz"]),
            "Drums": buscar_link_inteligente(res_api, ["drum", "bateria", "percussion"]),
            "Bass": buscar_link_inteligente(res_api, ["bass", "baixo"]),
            "Guitar": buscar_link_inteligente(res_api, ["guitar", "electric", "acoustic"]),
            "Other": buscar_link_inteligente(res_api, ["other", "rest", "piano", "synth"])
        }
        
        texturas = {
            "Vocals": tex_vocal, "Drums": tex_drums, 
            "Bass": tex_bass, "Guitar": tex_guitar, "Other": tex_other
        }

        mix_final, sr_final = None, 44100
        processou_algo = False

        print("\n--- 3. DOWNLOAD & EFEITOS ---")
        for inst, url_stem in urls_stems.items():
            if not url_stem: continue
            
            processou_algo = True
            print(f"⬇️ Baixando {inst}...")
            stem_path = baixar_stem(url_stem, f"temp_{inst}.wav")
            tex_path = texturas.get(inst)

            if tex_path:
                print(f"✨ Aplicando textura em {inst}...")
                audio, sr = aplicar_textura_local(stem_path, tex_path)
            else:
                print(f"🔹 Mantendo original: {inst}")
                audio, sr = librosa.load(stem_path, sr=44100)

            # Mixagem
            if mix_final is None:
                mix_final, sr_final = audio, sr
            else:
                m_len = min(len(mix_final), len(audio))
                mix_final = mix_final[:m_len] + audio[:m_len]

        if not processou_algo:
            return None

        # Salvar
        if mix_final is not None:
            mix_final = mix_final / np.max(np.abs(mix_final)) # Normalizar
            output = "resultado_final.wav"
            sf.write(output, mix_final, sr_final)
            print("✅ Concluído com sucesso!")
            return output
            
    except Exception as e:
        print(f"❌ ERRO GERAL: {e}")
        return None

# --- INTERFACE ---
with gr.Blocks(title="Texture Tool V2") as demo:
    gr.Markdown("# 🎹 Texture Transfer Tool")
    
    with gr.Row():
        input_music = gr.Audio(label="1. Música Original", type="filepath")
    
    gr.Markdown("### 2. Escolha as texturas (Deixe vazio para manter original)")
    with gr.Row():
        tex_vocal = gr.Audio(label="Textura Voz", type="filepath")
        tex_drums = gr.Audio(label="Textura Bateria", type="filepath")
        tex_bass = gr.Audio(label="Textura Baixo", type="filepath")
        tex_guitar = gr.Audio(label="Textura Guitarra", type="filepath")
        tex_other = gr.Audio(label="Textura Outros", type="filepath")

    btn = gr.Button("Processar", variant="primary")
    out = gr.Audio(label="Resultado Final")

    btn.click(
        fn=processar_tudo,
        inputs=[input_music, tex_vocal, tex_drums, tex_bass, tex_guitar, tex_other],
        outputs=out
    )

if __name__ == "__main__":
    demo.launch()