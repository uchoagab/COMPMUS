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
WORKFLOW_ID = os.getenv("WORKFLOW_ID")

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

# --- NOVA FUNÇÃO: REMIXAR (Manipulação do Resultado) ---
def remixar(vol_vocal, vol_drums, vol_bass, vol_guitar, vol_other, caminhos_dict):
    """
    Lê os arquivos já processados e cria um novo mix baseado nos sliders.
    Usa soundfile para leitura rápida e segura.
    """
    if not caminhos_dict:
        return None

    print(f"🎚️ Remixando: V:{vol_vocal}, D:{vol_drums}, B:{vol_bass}...")
    
    mix_final = None
    sr_final = 44100
    
    # Mapeamento Slider -> Chave do Dicionário
    config = [
        ("Vocals", vol_vocal),
        ("Drums", vol_drums),
        ("Bass", vol_bass),
        ("Guitar", vol_guitar),
        ("Other", vol_other)
    ]

    for inst, vol in config:
        path = caminhos_dict.get(inst)
        
        # Só processa se o arquivo existir e o volume for > 0
        if path and os.path.exists(path) and vol > 0:
            # Ler com Soundfile (muito mais rápido que librosa para apenas ler)
            try:
                data, sr = sf.read(path)
                
                # Garantir Mono se necessário
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                # Aplicar ganho
                y = data * vol
                
                if mix_final is None:
                    mix_final = y
                else:
                    # Somar arrays de tamanhos possivelmente diferentes (segurança)
                    m_len = min(len(mix_final), len(y))
                    mix_final = mix_final[:m_len] + y[:m_len]
            except Exception as e:
                print(f"Erro ao ler {path}: {e}")

    if mix_final is None:
        return None
        
    # Normalizar para evitar clipping no remix
    max_val = np.max(np.abs(mix_final))
    if max_val > 0:
        mix_final = mix_final / (max_val + 1e-9) * 0.95

    output_remix = "resultado_remix_final.wav"
    sf.write(output_remix, mix_final, sr_final)
    return output_remix

# --- PRINCIPAL ---
def processar_tudo(input_music, tex_vocal, tex_drums, tex_bass, tex_guitar, tex_other):
    try:
        print("\n--- 1. UPLOAD ---")
        if not input_music: return [None] * 7 # Retorna 7 Nones agora (6 audios + 1 estado)

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

        # Dicionário para armazenar caminhos dos stems processados
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

            # Mixagem Inicial (Padrão 1.0 volume)
            if mix_final is None:
                mix_final, sr_final = audio, sr
            else:
                m_len = min(len(mix_final), len(audio))
                mix_final = mix_final[:m_len] + audio[:m_len]

        if not processou_algo:
            return [None] * 7

        # Salvar mix final inicial
        output_completo = None
        if mix_final is not None:
            mix_final = mix_final / np.max(np.abs(mix_final)) # Normalizar
            output_completo = "resultado_completo.wav"
            sf.write(output_completo, mix_final, sr_final)
            print("✅ Concluído com sucesso!")
            
        # Retornar os outputs e o ESTADO (stems_processados)
        return (
            stems_processados.get("Vocals"),
            stems_processados.get("Drums"),
            stems_processados.get("Bass"),
            stems_processados.get("Guitar"),
            stems_processados.get("Other"),
            output_completo,
            stems_processados # << O segredo: passamos o dicionário para o State
        )
            
    except Exception as e:
        print(f"❌ ERRO GERAL: {e}")
        return [None] * 7

# --- INTERFACE ---
with gr.Blocks(title="Texture Tool V2 + Mixer") as demo:
    gr.Markdown("# 🎹 Texture Transfer Tool & Mixer")
    gr.Markdown("### Separe, texturize e depois manipule a mixagem final.")

    # ESTADO: Guarda os caminhos dos arquivos gerados para o remixer usar
    stems_state = gr.State({}) 
    
    with gr.Row():
        input_music = gr.Audio(label="1. Música Original", type="filepath")
    
    gr.Markdown("### 2. Escolha as texturas (Deixe vazio para manter original)")
    with gr.Row():
        tex_vocal = gr.Audio(label="Textura Voz", type="filepath")
        tex_drums = gr.Audio(label="Textura Bateria", type="filepath")
        tex_bass = gr.Audio(label="Textura Baixo", type="filepath")
        tex_guitar = gr.Audio(label="Textura Guitarra", type="filepath")
        tex_other = gr.Audio(label="Textura Outros", type="filepath")

    btn_process = gr.Button("🎵 Processar e Gerar Stems", variant="primary", size="lg")
    
    gr.Markdown("---")
    
    # SEÇÃO DO MIXER
    with gr.Row():
        # Coluna da Esquerda: Controles de Volume
        with gr.Column(scale=1):
            gr.Markdown("### 🎚️ Manipulação (Mixer)")
            vol_vocal = gr.Slider(0, 2, value=1, step=0.1, label="Volume Voz")
            vol_drums = gr.Slider(0, 2, value=1, step=0.1, label="Volume Bateria")
            vol_bass = gr.Slider(0, 2, value=1, step=0.1, label="Volume Baixo")
            vol_guitar = gr.Slider(0, 2, value=1, step=0.1, label="Volume Guitarra")
            vol_other = gr.Slider(0, 2, value=1, step=0.1, label="Volume Outros")
            
            btn_remix = gr.Button("🔄 Atualizar Mix", variant="secondary")

        # Coluna da Direita: Player Principal
        with gr.Column(scale=2):
            gr.Markdown("### 🎶 Mix Final (Resultado)")
            out_completo = gr.Audio(label="Resultado Final Manipulável", type="filepath")

    gr.Markdown("---")
    gr.Markdown("### 📂 Stems Individuais (Para conferência)")
    with gr.Row():
        out_vocal = gr.Audio(label="🎤 Vocal")
        out_drums = gr.Audio(label="🥁 Bateria")
        out_bass = gr.Audio(label="🎸 Baixo")
        out_guitar = gr.Audio(label="🎸 Guitarra")
        out_other = gr.Audio(label="🎹 Outros")
    
    # Clique do Processamento (Gera tudo pela primeira vez)
    btn_process.click(
        fn=processar_tudo,
        inputs=[input_music, tex_vocal, tex_drums, tex_bass, tex_guitar, tex_other],
        outputs=[out_vocal, out_drums, out_bass, out_guitar, out_other, out_completo, stems_state]
    )

    # Clique do Remix (Usa o Estado e os Sliders para gerar novo mix rápido)
    btn_remix.click(
        fn=remixar,
        inputs=[vol_vocal, vol_drums, vol_bass, vol_guitar, vol_other, stems_state],
        outputs=out_completo
    )

if __name__ == "__main__":
    demo.launch(share=True)