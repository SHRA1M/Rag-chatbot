import streamlit as st
import os
import re
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# --- 1. PAGE CONFIG (Must be first) ---
st.set_page_config(
    page_title="DP Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    /* MOBILE FIX: Force everything to have proper contrast */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* All chat messages - base styling */
    [data-testid="stChatMessage"] {
        background-color: #f0f2f6 !important;
        color: #000000 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        margin: 8px 0 !important;
    }
    
    /* ============================================
       USER MESSAGES - Hide avatar, clean bubble
       ============================================ */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #002147 !important;
        color: #ffffff !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        max-width: 80% !important;
        flex-direction: row-reverse !important;
    }
    
    /* HIDE THE UGLY ORANGE USER AVATAR */
    [data-testid="stChatMessageAvatarUser"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Also hide the avatar container for user */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:first-child {
        display: none !important;
    }
    
    /* User message text - white */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
        color: #ffffff !important;
    }
    
    /* ============================================
       ASSISTANT MESSAGES
       ============================================ */
    [data-testid="stChatMessage"]:has(img) {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e0e0e0 !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        max-width: 80% !important;
    }
    
    /* Assistant message text - black */
    [data-testid="stChatMessage"]:has(img) p {
        color: #000000 !important;
    }
    
    /* Fix text color inheritance for ALL paragraphs */
    [data-testid="stChatMessage"] p {
        color: inherit !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    
    /* ============================================
       ARABIC TEXT HANDLING (Desktop & Mobile)
       ============================================ */
    .arabic-text {
        direction: rtl !important;
        text-align: right !important;
        color: inherit !important;
        unicode-bidi: plaintext !important;
        font-family: 'Segoe UI', 'Arial', 'Tahoma', sans-serif !important;
    }
    
    /* Auto-detect Arabic in user messages and align RTL */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p:lang(ar),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p[dir="rtl"] {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* Mobile-specific Arabic fixes */
    @media (max-width: 768px) {
        .arabic-text {
            direction: rtl !important;
            text-align: right !important;
            unicode-bidi: embed !important;
        }
        
        /* Ensure chat messages don't break on mobile */
        [data-testid="stChatMessage"] {
            max-width: 90% !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        
        /* Better Arabic rendering on mobile */
        [data-testid="stChatMessage"] p {
            font-size: 15px !important;
            line-height: 1.6 !important;
        }
    }
    
    /* ============================================
       INPUT BOX STYLING
       ============================================ */
    /* Target the container that holds the input */
    .stChatInput > div {
        background-color: #ffffff !important; /* Force white background to match input */
        border-color: #cccccc !important;
        border-radius: 20px !important;
        box-shadow: none !important; /* Kill default shadow */
    }

    /* Target focus state specifically to kill orange glow */
    .stChatInput > div:focus-within {
        border-color: #cccccc !important; /* Keep it grey on focus */
        box-shadow: 0 0 0 1px #cccccc !important; /* Optional: subtle grey glow instead of orange */
    }

    .stChatInput input, .stChatInput textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        caret-color: #000000 !important; /* Black cursor */
    }

    /* FORCE SEND BUTTON COLOR (Kill the Orange) */
    .stChatInput button {
        background-color: transparent !important;
        color: #002147 !important; /* Dark Blue Icon */
        border: none !important;
    }
    
    .stChatInput button:hover {
        background-color: #f0f2f6 !important;
        color: #002147 !important;
    }
    
    /* Support RTL input for Arabic */
    .stChatInput input:lang(ar),
    .stChatInput input[dir="rtl"] {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* ============================================
       HIDE STREAMLIT BRANDING
       ============================================ */
    #MainMenu, header, footer, .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }
    
    .stApp > header {
        display: none !important;
    }
    
    [data-testid="stDecoration"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] { 
        display: none !important; 
    }
    </style>
""", unsafe_allow_html=True)

# --- 2.5. AGGRESSIVE FOOTER REMOVAL (JavaScript + CSS) ---
st.html("""
<style>
    footer, [data-testid="stFooter"], .stFooter, 
    div[class*="footer"], a[href*="streamlit.io"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
</style>
<script>
    // Remove footer on load and keep checking
    function removeFooter() {
        const selectors = [
            'footer',
            '[data-testid="stFooter"]',
            '.stFooter',
            'a[href*="streamlit.io"]'
        ];
        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => el.remove());
        });
    }
    
    // Run immediately
    removeFooter();
    
    // Run after DOM is ready
    document.addEventListener('DOMContentLoaded', removeFooter);
    
    // Keep checking every 500ms for dynamically added footers
    setInterval(removeFooter, 500);
</script>
""")

# --- 3. LOAD KNOWLEDGE BASE ---
@st.cache_resource
def load_retriever():
    try:
        # Check if we need to unzip the index (for Cloud Deployment)
        if not os.path.exists("faiss_index") and os.path.exists("faiss_index.zip"):
            print("Unzipping faiss_index.zip...")
            import zipfile
            with zipfile.ZipFile("faiss_index.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Unzipping complete.")
            
        embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4}), None
    except Exception as e:
        return None, str(e)


retriever, retriever_error = load_retriever()

# --- 4. PATHS ---
logo_path = "data/logo_transparent.png"
if not os.path.exists(logo_path):
    logo_path = None

# --- 5. GROQ CLIENT ---
client = None
api_error = None
try:
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if api_key:
        client = Groq(api_key=api_key)
    else:
        api_error = "GROQ_API_KEY not found in secrets"
except Exception as e:
    api_error = str(e)

# --- 6. MODEL CONFIGURATION ---
GROQ_MODEL = "llama-3.3-70b-versatile"
BACKUP_MODEL = "llama-3.1-8b-instant"  # Faster backup model

# --- 7. GREETINGS ---
GREETING_EN = """Hello! Welcome to **Digital Protection**.

I am here to help you with your questions.

How can I help you?"""

GREETING_AR = """<div class="arabic-text">

مرحبا! اهلا بك في **Digital Protection**.

انا هنا لمساعدتك في اسئلتك.

كيف يمكنني مساعدتك؟

</div>"""

# --- 8. SYSTEM INSTRUCTIONS ---
SYSTEM_INSTRUCTIONS_EN = """You are DP Assistant for Digital Protection, a data protection consultancy in Amman, Jordan.

LANGUAGE: Respond in ENGLISH only.

RULES:
1. NO EMOJIS ever
2. NO LEGAL ADVICE - say "I cannot provide legal advice. Please consult a qualified legal professional."
3. NO CONTRACTS - say "I cannot generate contracts. Please contact our team."
4. NO SPECIFIC PRICES - say pricing depends on scope
5. NO IT SUPPORT for printers, WiFi, hardware

STYLE: Give complete, helpful answers. Use bullet points for lists. Professional but friendly.

SERVICES:
- Privacy & Compliance: GDPR, ISO 27701, CBJ
- Security Assessments: Vulnerability scanning, risk analysis
- Network Security: Firewalls, WAF
- Identity & Access Management: IAM/PAM

CONTACT: info@dp-technologies.net | +962 790 552 879 | Amman, Jordan"""

SYSTEM_INSTRUCTIONS_AR = """انت مساعد DP لشركة Digital Protection في عمان، الاردن.

اللغة: رد بالعربية فقط.

القواعد:
1. بدون رموز تعبيرية ابدا
2. بدون استشارات قانونية - قل "لا استطيع تقديم استشارات قانونية. يرجى استشارة محام مختص."
3. بدون عقود - قل "لا استطيع انشاء عقود. يرجى التواصل مع فريقنا."
4. بدون اسعار محددة - قل التسعير يعتمد على نطاق المشروع
5. بدون دعم تقني للطابعات والواي فاي

الاسلوب: قدم اجابات كاملة ومفيدة. استخدم النقاط للقوائم. مهني وودود.

الخدمات:
- الخصوصية والامتثال: GDPR، ISO 27701، البنك المركزي الاردني
- تقييمات الامن: فحص الثغرات، تحليل المخاطر
- امن الشبكات: جدران الحماية، WAF
- ادارة الهوية والوصول: IAM/PAM

التواصل: info@dp-technologies.net | +962 790 552 879 | عمان، الاردن"""

# --- 9. HELPER FUNCTIONS ---
def is_arabic(text):
    """Detect if text contains Arabic characters"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def clean_response(answer, is_arabic_response=False):
    """Clean up the response text"""
    # Remove robotic labels
    labels = ["Direct answer:", "Key Points:", "Key Considerations:", "Next Step:", 
              "Response:", "Answer:", "الاجابة:", "النقاط الرئيسية:", "الخطوة التالية:"]
    for label in labels:
        answer = answer.replace(label, "")
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    answer = emoji_pattern.sub('', answer)
    
    # Clean whitespace
    while "\n\n\n" in answer:
        answer = answer.replace("\n\n\n", "\n\n")
    
    answer = answer.strip()
    
    # Wrap Arabic in RTL div
    if is_arabic_response:
        answer = f'<div class="arabic-text">{answer}</div>'
    
    return answer

def get_fallback_response(prompt, is_arabic_lang):
    """Get a fallback response when API fails"""
    prompt_lower = prompt.lower()
    fallback = FALLBACK_AR if is_arabic_lang else FALLBACK_EN
    
    if any(word in prompt_lower for word in ["service", "خدم", "offer", "تقدم"]):
        return fallback["services"]
    elif any(word in prompt_lower for word in ["price", "cost", "سعر", "تكلف", "كم"]):
        return fallback["pricing"]
    elif any(word in prompt_lower for word in ["where", "location", "اين", "موقع"]):
        return fallback["location"]
    else:
        return fallback["default"]

# --- 10. FALLBACK RESPONSES ---
FALLBACK_EN = {
    "services": "We offer cybersecurity and compliance services including GDPR, ISO 27701, CBJ compliance, security assessments, and identity management. Contact us at info@dp-technologies.net for details.",
    "pricing": "Pricing depends on the scope of your project. We offer fixed-price, time and materials, and retainer options. Contact info@dp-technologies.net for a quote.",
    "location": "We are located in Amman, Jordan. Contact us at info@dp-technologies.net or +962 790 552 879.",
    "default": "Thank you for your message. For detailed assistance, please contact our team at info@dp-technologies.net or +962 790 552 879."
}

FALLBACK_AR = {
    "services": "نقدم خدمات الامن السيبراني والامتثال بما في ذلك GDPR و ISO 27701 والبنك المركزي الاردني وتقييمات الامن. تواصل معنا على info@dp-technologies.net",
    "pricing": "التسعير يعتمد على نطاق مشروعك. نقدم خيارات السعر الثابت والوقت والمواد والاشتراك. تواصل معنا للحصول على عرض سعر.",
    "location": "نحن في عمان، الاردن. تواصل معنا على info@dp-technologies.net او +962 790 552 879",
    "default": "شكرا لرسالتك. للمساعدة التفصيلية، يرجى التواصل مع فريقنا على info@dp-technologies.net او +962 790 552 879"
}

# --- 11. INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"

if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False

if "error_count" not in st.session_state:
    st.session_state.error_count = 0

# --- 12. HEADER WITH LANGUAGE TOGGLE ---
query_params = st.query_params
is_embedded = query_params.get("embed", "false").lower() == "true"

if not is_embedded:
    col1, col2, col3 = st.columns([1, 4, 2])
    with col1:
        if logo_path:
            st.image(logo_path, width=50)
    with col2:
        st.markdown("### Digital Protection Support")
    with col3:
        if st.session_state.ui_language == "en":
            if st.button("بالعربية", key="lang_toggle"):
                st.session_state.ui_language = "ar"
                st.session_state.messages = []
                st.session_state.greeting_shown = False
                st.rerun()
        else:
            if st.button("English", key="lang_toggle"):
                st.session_state.ui_language = "en"
                st.session_state.messages = []
                st.session_state.greeting_shown = False
                st.rerun()
else:
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.session_state.ui_language == "en":
            if st.button("عربي", key="lang_toggle_embed"):
                st.session_state.ui_language = "ar"
                st.session_state.messages = []
                st.session_state.greeting_shown = False
                st.rerun()
        else:
            if st.button("EN", key="lang_toggle_embed"):
                st.session_state.ui_language = "en"
                st.session_state.messages = []
                st.session_state.greeting_shown = False
                st.rerun()

# --- 13. SHOW GREETING ---
if not st.session_state.greeting_shown:
    if st.session_state.ui_language == "ar":
        st.session_state.messages = [{"role": "assistant", "content": GREETING_AR}]
    else:
        st.session_state.messages = [{"role": "assistant", "content": GREETING_EN}]
    st.session_state.greeting_shown = True

# --- 14. DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    avatar = logo_path if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)

# --- 15. CHAT INPUT ---
input_placeholder = "اكتب رسالتك..." if st.session_state.ui_language == "ar" else "Type your message..."

if prompt := st.chat_input(input_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar=logo_path):
        response_placeholder = st.empty()
        
        # 1. Search knowledge base
        context = ""
        if retriever:
            try:
                # ENHANCED RAG: Include recent chat history
                chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
                enhanced_prompt = f"Chat History: {chat_history}\nCurrent Question: {prompt}"
                
                search_results = retriever.invoke(enhanced_prompt)
                context = "\n".join([doc.page_content for doc in search_results])
            except Exception as e:
                print(f"DEBUG: Retriever failed: {e}")
        
        # 2. Check language
        user_is_ar = is_arabic(prompt) or st.session_state.ui_language == "ar"
        system_prompt = SYSTEM_INSTRUCTIONS_AR if user_is_ar else SYSTEM_INSTRUCTIONS_EN

        # 3. Call API with Fallback Logic
        stream = None
        used_model = GROQ_MODEL
        
        try:
            # Prepare Enforced Prompt (Guardrails)
            enforced_prompt = f"""Answer the question using ONLY the information in the CONTEXT below.

RULES:
1. NO emojis
2. Give a COMPLETE answer using the context - include all relevant details
3. Only say "contact us" for pricing questions or if info is truly not in context
4. Answer in {'Arabic' if user_is_ar else 'English'} only

CONTEXT:
{context}

QUESTION:
{prompt}

ANSWER:"""

            # Build Message Chain
            api_messages = []
            for m in st.session_state.messages[-5:]:
                 api_messages.append({"role": m["role"], "content": m["content"]})
            
            api_messages.append({"role": "user", "content": enforced_prompt})

            # Add System Instructions
            system_reinforcement = system_prompt + "\n\nREMEMBER: Answer using the context provided. No emojis."
            api_messages.append({"role": "system", "content": system_reinforcement})

            # Try Primary Model
            stream = client.chat.completions.create(
                messages=api_messages,
                model=GROQ_MODEL,
                temperature=0.1,
                stream=True,
            )
        except Exception as e:
            print(f"Primary model failed: {e}")
            # Try Backup Model
            try:
                stream = client.chat.completions.create(
                    messages=api_messages,
                    model=BACKUP_MODEL,
                    temperature=0.1,
                    stream=True,
                )
                used_model = BACKUP_MODEL
            except Exception as e2:
                print(f"Backup model also failed: {e2}")
                stream = None

        # 4. Process Stream or Show Static Fallback
        if stream:
            try:
                full_response = ""
                last_update_time = time.time()
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        current_time = time.time()
                        if current_time - last_update_time > 0.05:
                            display_text = clean_response(full_response, user_is_ar)
                            response_placeholder.markdown(display_text + "▌", unsafe_allow_html=True)
                            last_update_time = current_time
                
                final_answer = clean_response(full_response, user_is_ar)
                response_placeholder.markdown(final_answer, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            except Exception as e:
                 print(f"Stream processing error: {e}")
                 fallback = get_fallback_response(prompt, user_is_ar)
                 response_placeholder.markdown(fallback)
        else:
            # If both models failed
            fallback = get_fallback_response(prompt, user_is_ar)
            response_placeholder.markdown(fallback)