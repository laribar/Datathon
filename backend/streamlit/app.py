# ============================================
# >>> FILE: backend/streamlit/app.py
# ============================================
import uuid
from pathlib import Path
import streamlit as st
from streamlit.components.v1 import html

st.set_page_config(page_title="Sala de Entrevista", page_icon="🎥", layout="wide")
st.title("🎥 Sala de Entrevista (vídeo + emoção)")

# ---- Inputs principais
c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
with c0:
    room = st.text_input("Nome da sala", value="entrevista-demo")
with c1:
    display_name = st.text_input("Seu nome na call", value=f"User-{str(uuid.uuid4())[:5]}")
with c2:
    role = st.selectbox("Seu papel", ["candidato", "entrevistador"], index=0)
with c3:
    participant_id = st.text_input("ID participante (logs)", value=f"p-{str(uuid.uuid4())[:4]}")

st.divider()
left, right = st.columns([2, 1])

# ---- Esquerda: Jitsi
with left:
    st.subheader("Entrevista em Vídeo")
    jitsi_html = f"""
    <div id="meet"></div>
    <script src="https://meet.jit.si/external_api.js"></script>
    <script>
      const domain = "meet.jit.si";
      const options = {{
        roomName: "{room}",
        width: "100%",
        height: 620,
        parentNode: document.querySelector('#meet'),
        userInfo: {{ displayName: "{display_name}" }},
        configOverwrite: {{ prejoinPageEnabled: true }},
        interfaceConfigOverwrite: {{
          SHOW_JITSI_WATERMARK: false,
          SHOW_WATERMARK_FOR_GUESTS: false
        }}
      }};
      const api = new JitsiMeetExternalAPI(domain, options);
    </script>
    """
    html(jitsi_html, height=640)

# ---- Direita: seu teste.html embutido
with right:
    st.subheader("Análise de Emoção (seu teste.html)")
    backend_default = "https://SEU-APP.onrender.com/api/emotion"  # troque para sua URL real do Render
    backend_url = st.text_input("Backend /api/emotion", value=backend_default)

    html_path = Path(__file__).parent / "teste.html"
    if not html_path.exists():
        st.error(f"Arquivo não encontrado: {html_path}")
    else:
        content = html_path.read_text(encoding="utf-8")

        # 1) injeta variáveis globais para o teste.html (ROOM/ROLE/PARTICIPANT)
        inject = (
            f"<script>"
            f"window.EMBED_ROOM={room!r};"
            f"window.EMBED_PARTICIPANT={participant_id!r};"
            f"window.EMBED_ROLE={role!r};"
            f"</script>"
        )
        if "<script>" in content:
            content = content.replace("<script>", inject + "\n<script>", 1)
        else:
            content = inject + content

        # 2) troca a URL do backend local pela do Render (campo acima)
        content = content.replace("http://127.0.0.1:8000/api/emotion", backend_url)

        # 3) entrega o HTML inteiro renderizado
        html(content, height=760, scrolling=True)
        st.caption("Clique em **Iniciar** no painel para enviar frames ao backend.")
# ============================================
# <<< END FILE
# ============================================
