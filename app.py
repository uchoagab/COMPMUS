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
    """Aplica textura usando phase vocoder para mixagem fluída que preserva características tonais."""
    if not stem_path or not textura_path: return None, 44100
    
    y_stem, sr = librosa.load(stem_path, sr=44100)
    y_textura, _ = librosa.load(textura_path, sr=44100)

    # Loop da textura
    if len(y_textura) < len(y_stem):
        reps = int(np.ceil(len(y_stem) / len(y_textura)))
        y_textura = np.tile(y_textura, reps)
    y_textura = y_textura[:len(y_stem)]

    # STFT para análise espectral
    n_fft = 2048
    hop_length = 512
    
    # Espectrogramas
    D_stem = librosa.stft(y_stem, n_fft=n_fft, hop_length=hop_length)
    D_textura = librosa.stft(y_textura, n_fft=n_fft, hop_length=hop_length)
    
    # Magnitude e fase
    mag_stem = np.abs(D_stem)
    phase_stem = np.angle(D_stem)
    mag_textura = np.abs(D_textura)
    
    # Envelope dinâmico do stem original
    env_stem = librosa.feature.rms(y=y_stem, frame_length=n_fft, hop_length=hop_length)[0]
    env_stem = env_stem / (np.max(env_stem) + 1e-9)
    
    # Transferência espectral: usa magnitude da textura modulada pela dinâmica do stem
    # e mantém a fase do stem original para preservar características tonais
    mag_textura_ajustada = mag_textura * env_stem
    
    # Mistura ponderada: 70% textura + 30% stem original para preservar harmônicos
    mag_final = 0.7 * mag_textura_ajustada + 0.3 * mag_stem
    
    # Reconstrução com fase do stem original
    D_final = mag_final * np.exp(1j * phase_stem)
    
    # ISTFT para obter áudio resultante
    y_resultado = librosa.istft(D_final, hop_length=hop_length, length=len(y_stem))
    
    # Normalização suave
    y_resultado = y_resultado / (np.max(np.abs(y_resultado)) + 1e-9) * 0.8
    
    return y_resultado, sr

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

        # Dicionário para armazenar stems processados
        stems_processados = {}
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

            # Salvar stem individual processado
            output_individual = f"resultado_{inst}.wav"
            sf.write(output_individual, audio, sr)
            stems_processados[inst] = output_individual

            # Mixagem
            if mix_final is None:
                mix_final, sr_final = audio, sr
            else:
                m_len = min(len(mix_final), len(audio))
                mix_final = mix_final[:m_len] + audio[:m_len]

        if not processou_algo:
            return None, None, None, None, None, None

        # Salvar mix final
        if mix_final is not None:
            mix_final = mix_final / np.max(np.abs(mix_final)) # Normalizar
            output_completo = "resultado_completo.wav"
            sf.write(output_completo, mix_final, sr_final)
            print("✅ Concluído com sucesso!")
            
            # Retornar os 6 outputs: vocal, drums, bass, guitar, other, completo
            return (
                stems_processados.get("Vocals"),
                stems_processados.get("Drums"),
                stems_processados.get("Bass"),
                stems_processados.get("Guitar"),
                stems_processados.get("Other"),
                output_completo
            )
            
    except Exception as e:
        print(f"❌ ERRO GERAL: {e}")
        return None, None, None, None, None, None

# --- INTERFACE ---
with gr.Blocks(title="Texture Tool V2") as demo:
    gr.Markdown("# 🎹 Texture Transfer Tool")
    gr.Markdown("### Separe sua música em stems e aplique texturas personalizadas")
    
    with gr.Row():
        input_music = gr.Audio(label="1. Música Original", type="filepath")
    
    gr.Markdown("### 2. Escolha as texturas (Deixe vazio para manter original)")
    with gr.Row():
        tex_vocal = gr.Audio(label="Textura Voz", type="filepath")
        tex_drums = gr.Audio(label="Textura Bateria", type="filepath")
    with gr.Row():
        tex_bass = gr.Audio(label="Textura Baixo", type="filepath")
        tex_guitar = gr.Audio(label="Textura Guitarra", type="filepath")
        tex_other = gr.Audio(label="Textura Outros", type="filepath")

    btn = gr.Button("🎵 Processar e Gerar Stems", variant="primary", size="lg")
    
    gr.Markdown("### 3. Resultados - Stems Individuais Modificados")
    with gr.Row():
        out_vocal = gr.Audio(label="🎤 Vocal Modificado")
        out_drums = gr.Audio(label="🥁 Bateria Modificada")
    with gr.Row():
        out_bass = gr.Audio(label="🎸 Baixo Modificado")
        out_guitar = gr.Audio(label="🎸 Guitarra Modificada")
    with gr.Row():
        out_other = gr.Audio(label="🎹 Outros Modificados")
    
    gr.Markdown("### 4. Mix Final Completo")
    out_completo = gr.Audio(label="🎶 Música Completa Modificada")

    btn.click(
        fn=processar_tudo,
        inputs=[input_music, tex_vocal, tex_drums, tex_bass, tex_guitar, tex_other],
        outputs=[out_vocal, out_drums, out_bass, out_guitar, out_other, out_completo]
    )

if __name__ == "__main__":
    demo.launch(share=True)
