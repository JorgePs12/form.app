import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path
import json
import os

# --- charts ---
import plotly.express as px

CSV_PATH = Path("avaliacao_facial.csv")

# --------------------------
# 1) FORM (your dictionary)
# --------------------------
FORM = {
    "Dados Pessoais": [
        {"key": "nome", "label": "Nome", "type": "text", "required": True},
        {"key": "data", "label": "Data da avaliação", "type": "date", "required": True},
        {"key": "nascimento", "label": "Data de nascimento", "type": "date", "required": True},
        {"key": "idade", "label": "Idade", "type": "number", "min": 0, "max": 130, "required": True},
        {"key": "sexo", "label": "Sexo", "type": "select", "options": ["", "F", "M", "Outro"], "required": True},
        {"key": "fone", "label": "Fone", "type": "text"},
        {"key": "endereco", "label": "Endereço", "type": "text"},
        {"key": "profissao", "label": "Profissão", "type": "text"},
    ],
    "Queixa": [
        {"key": "queixa_principal", "label": "Queixa principal", "type": "textarea"},
        {"key": "objetivo", "label": "Objetivo do cliente", "type": "textarea"},
        {"key": "historia_queixa", "label": "História da patologia (queixa)", "type": "textarea"},
        {"key": "tratamentos_anteriores", "label": "Já realizou tratamentos anteriores?", "type": "yn"},
        {"key": "tratamentos_anteriores_quais", "label": "Quais", "type": "textarea",
         "show_if": ("tratamentos_anteriores", "Sim")},
        {"key": "grau_satisfacao", "label": "Grau de satisfação", "type": "text",
         "show_if": ("tratamentos_anteriores", "Sim")},
    ],
    "Antecedentes": [
        {"key": "alergias", "label": "Antecedentes alérgicos?", "type": "yn"},
        {"key": "alergias_quais", "label": "Quais alergias?", "type": "text",
         "show_if": ("alergias", "Sim")},
        {"key": "herpes_ativo", "label": "Herpes ativo?", "type": "yn"},
        {"key": "diabetes", "label": "Diabetes descompensada?", "type": "yn"},
        {"key": "cancer_pele", "label": "Histórico de câncer de pele?", "type": "yn"},
        {"key": "fumante", "label": "Fumante?", "type": "yn"},
        {"key": "fumante_qtd", "label": "Quantidade por dia", "type": "text",
         "show_if": ("fumante", "Sim")},
        {"key": "cosmeticos", "label": "Faz uso de cosméticos?", "type": "yn"},
        {"key": "cosmeticos_quais", "label": "Quais cosméticos?", "type": "textarea",
         "show_if": ("cosmeticos", "Sim")},
        {"key": "protetor_solar", "label": "Usa protetor solar?", "type": "yn"},
        {"key": "protetor_solar_frequencia", "label": "Quantos dias por semana você usa?", "type": "number",
         "min": 0, "max": 7, "show_if": ("protetor_solar", "Sim")},
        {"key": "agua", "label": "Ingestão diária de água", "type": "select",
         "options": ["", "Menos de 1 litro", "1-2 litros", "Mais de 2 litros"]},
        {"key": "intestino", "label": "Funcionamento do intestino:", "type": "textarea"},
        {"key": "digestao", "label": "Problemas de digestão:", "type": "select",
         "options": ["", "Normal", "Azia", "Refluxo", "Gastrite"]},
        {"key": "sono", "label": "Qualidade do sono", "type": "select",
         "options": ["", "Acorda disposto", "Acorda cansado"]},
        {"key": "dores", "label": "Sente dores no corpo?", "type": "yn"},

        {"key": "dores_quais", "label": "Quais dores?", "type": "options",
         "options": ["", "Articulação", "Muscular"], "show_if": ("dores", "Sim")},
        {"key": "humor", "label": "Como é seu humor", "type": "options",
         "options": ["", "Bom", "Regular", "Irritado", "Estressado", "Nervoso", "Ansioso", "Triste"]},
        {"key": "imunidade", "label": "Como é sua imunidade", "type": "options",
         "options": ["", "Boa", "Fica doente com frequência"]},
        {"key": "problema_respiratorio", "label": "Apresenta algum problema respiratório", "type": "options",
         "options": ["", "Rinite", "Sinusite"]},

        {"key": "dores_de_cabeca", "label": "Tem dores de cabeça?", "type": "yn"},
        {"key": "frequencia_dores_cabeca", "label": "Frequência semanal de dores de cabeça", "type": "number",
         "min": 0, "max": 7, "show_if": ("dores_de_cabeca", "Sim")},
        {"key": "cabelo_pele_unhas", "label": "Como estão cabelo, pele e unhas", "type": "textarea"},
        {"key": "atividade_fisica", "label": "Faz atividade física?", "type": "yn"},
        {"key": "atividade_fisica_qual", "label": "Qual atividade física?", "type": "text",
         "show_if": ("atividade_fisica", "Sim")},
        {"key": "frequencia_atividade_fisica", "label": "Frequência semanal", "type": "number",
         "min": 0, "max": 7, "show_if": ("atividade_fisica", "Sim")},
        {"key": "expor_ao_sol", "label": "Costuma se expor ao sol?", "type": "yn"},
        {"key": "expor_ao_sol_frequencia", "label": "Com que frequência?", "type": "number",
         "min": 0, "max": 7, "show_if": ("expor_ao_sol", "Sim")},
        {"key": "acidos_e_despigmentantes", "label": "Faz uso recente de ácidos ou despigmentantes?", "type": "textarea"},
        {"key": "patologias_cutaneas", "label": "Possui alguma patologia cutânea?", "type": "yn"},
        {"key": "patologias_cutaneas_quais", "label": "Quais?", "type": "options",
         "options": ["", "Acne", "Rosácea", "Eczema", "Psoríase", "Vitiligo", "Lúpus"],
         "show_if": ("patologias_cutaneas", "Sim")},
        {"key": "inflamatorio_cutaneos", "label": "Processos inflamatórios cutâneos", "type": "textarea"},
        {"key": "cicatrizacao", "label": "Distúrbios de cicatrização", "type": "yn"},
        {"key": "antiflamatorios", "label": "Uso de anti inflamatórios", "type": "yn"},
        {"key": "depilacao", "label": "Depilação na face recentemente:", "type": "yn"},
        {"key": "tratamento_medico", "label": "Faz algum tratamento médico?", "type": "yn"},
        {"key": "qual_tratamento_medico", "label": "Quais tratamento médico?", "type": "textarea",
         "show_if": ("tratamento_medico", "Sim")},
        {"key": "medicamentos", "label": "Usa algum medicamento?", "type": "yn"},
        {"key": "quais_medicamentos", "label": "Quais medicamentos?", "type": "textarea",
         "show_if": ("medicamentos", "Sim")},
        {"key": "gestante", "label": "Está gestante ou amamentando?", "type": "yn"},
        {"key": "ciclo_mestrual", "label": "Ciclo menstrual regular?", "type": "yn"},
        {"key": "nao_regular", "label": "Observações do ciclo menstrual irregular", "type": "textarea",
         "show_if": ("ciclo_mestrual", "Não")},
        {"key": "marcapasso", "label": "Portador de marcapasso?", "type": "yn"},
        {"key": "problema_cardiovascular", "label": "Tem algum problema cardiovascular?", "type": "yn"},
        {"key": "quais_problemas_cardiovasculares", "label": "Quais problemas cardiovasculares?", "type": "textarea",
         "show_if": ("problema_cardiovascular", "Sim")},
        {"key": "problema_neurologico", "label": "Tem algum problema neurológico?", "type": "yn"},
        {"key": "quais_problemas_neurologicos", "label": "Quais problemas neurológicos?", "type": "textarea",
         "show_if": ("problema_neurologico", "Sim")},
        {"key": "problema_osteomuscular", "label": "Tem algum problema osteomuscular?", "type": "yn"},
        {"key": "quais_problemas_osteomusculares", "label": "Quais problemas osteomusculares?", "type": "textarea",
         "show_if": ("problema_osteomuscular", "Sim")},
        {"key": "protese_metalica", "label": "Presença de prótese metálica?", "type": "yn"},
        {"key": "local_protese_metalica", "label": "Local da prótese metálica", "type": "textarea",
         "show_if": ("protese_metalica", "Sim")},
        {"key": "hipertensao", "label": "Tem hipertensão?", "type": "yn"},
        {"key": "cirurgia_plastica", "label": "Cirurgia plástica?", "type": "yn"},
        {"key": "qual_cirurgia_plastica", "label": "Qual(is) cirurgia(s)?", "type": "textarea",
         "show_if": ("cirurgia_plastica", "Sim")},
        {"key": "quando_cirurgia_plastica", "label": "Quando foi realizada?", "type": "date",
         "show_if": ("cirurgia_plastica", "Sim")},
        {"key": "regiao_cirurgia_plastica", "label": "Região da cirurgia", "type": "textarea",
         "show_if": ("cirurgia_plastica", "Sim")},
        {"key": "edema", "label": "Presença de edema?", "type": "yn"},
        {"key": "localizacao_edema", "label": "Localização do edema", "type": "textarea",
         "show_if": ("edema", "Sim")},
    ],
    "Exame físico": [
        {"key": "biotipo", "label": "Biotipo Cutâneo", "type": "select",
         "options": ["", "normal", "mista", "oleosa", "seca"]},
        {"key": "fototipo", "label": "Fototipo Cutâneo", "type": "select",
         "options": ["", "I", "II", "III", "IV", "V"]},
        {"key": "hidratacao", "label": "Grau de hidratação", "type": "select",
         "options": ["", "desidratada", "semi hidratada", "hidratada"]},
        {"key": "pigmentacao", "label": "Pigmentação", "type": "multiselect",
         "options": ["hipocromias", "hipercromias"]},
        {"key": "pigmentacao_local", "label": "Local (pigmentação)", "type": "text"},
        {"key": "tipo_mancha", "label": "Tipo de mancha", "type": "text"},
        {"key": "rugas", "label": "Rugas", "type": "select",
         "options": ["", "superficial", "médias", "profundas"]},
        {"key": "rugas_local", "label": "Local (rugas)", "type": "text"},
        {"key": "flacidez_grau", "label": "Flacidez", "type": "select",
         "options": ["", "grau 1", "grau 2", "grau 3"]},
        {"key": "flacidez_local", "label": "Local (flacidez)", "type": "text"},
    ],
}

# --------------------------
# 2) Flatten + validation + Column Order
# --------------------------
def flatten_form(form_dict):
    flat = []
    for section, fields in form_dict.items():
        for f in fields:
            f2 = dict(f)
            f2["_section"] = section
            f2["_widget_key"] = f"{section}__{f['key']}"  # stable and unique
            flat.append(f2)
    return flat

FLAT_FIELDS = flatten_form(FORM)
KEY2WKEY = {f["key"]: f["_widget_key"] for f in FLAT_FIELDS}

def validate_form_fields(fields):
    problems = []
    keys = set()
    for f in fields:
        k = f.get("key")
        if not k:
            problems.append("A field is missing 'key'.")
            continue
        if k in keys:
            problems.append(f"Duplicate key found: '{k}'. Keys must be unique.")
        keys.add(k)

        t = f.get("type")
        if t in ("select", "multiselect", "options") and "options" not in f:
            problems.append(f"Field '{k}' type '{t}' needs an 'options' list.")

        if "show_if" in f:
            si = f["show_if"]
            if not (isinstance(si, tuple) and len(si) == 2):
                problems.append(f"Field '{k}' has invalid show_if. Use ('parent_key','value').")
            else:
                dep_key, _ = si
                if dep_key not in KEY2WKEY:
                    problems.append(f"Field '{k}' show_if references unknown key '{dep_key}'.")

    return problems

def build_column_order(fields):
    ordered = []
    added = set()

    for f in fields:
        k = f["key"]
        if "show_if" not in f and k not in added:
            ordered.append(k)
            added.add(k)

    for f in fields:
        if "show_if" in f:
            k = f["key"]
            dep_key, _ = f["show_if"]
            if k in added:
                continue
            if dep_key in ordered:
                i = ordered.index(dep_key)
                ordered.insert(i + 1, k)
            else:
                ordered.append(k)
            added.add(k)

    for f in fields:
        k = f["key"]
        if k not in added:
            ordered.append(k)
            added.add(k)

    return ordered

CSV_COLUMNS = build_column_order(FLAT_FIELDS)

# --------------------------
# 3) Auto-calculate age from birth date
# --------------------------
def calculate_age(birth_date):
    if not birth_date or not isinstance(birth_date, date):
        return None
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def update_age_from_birthdate():
    nascimento_wkey = "Dados Pessoais__nascimento"
    idade_wkey = "Dados Pessoais__idade"

    birth_date = st.session_state.get(nascimento_wkey)
    if birth_date and isinstance(birth_date, date):
        calculated_age = calculate_age(birth_date)
        if calculated_age is not None:
            st.session_state[idade_wkey] = calculated_age

# --------------------------
# 4) Rendering (show_if reads widget state directly)
# --------------------------
def should_show_field(f) -> bool:
    if "show_if" not in f:
        return True
    dep_key, dep_val = f["show_if"]
    dep_wkey = KEY2WKEY.get(dep_key)
    current = st.session_state.get(dep_wkey, None)
    return current == dep_val

def render_field(f):
    ftype = f.get("type", "text")
    key = f["key"]
    label = f.get("label", key)
    wkey = f["_widget_key"]
    is_required = f.get("required", False)

    if is_required:
        label = f"{label} *"

    if not should_show_field(f):
        return

    if ftype == "options":
        ftype = "select"

    if ftype == "text":
        st.text_input(label, key=wkey)

    elif ftype == "textarea":
        st.text_area(label, height=90, key=wkey)

    elif ftype == "number":
        minv = int(f.get("min", 0))
        maxv = int(f.get("max", 999))

        cur = st.session_state.get(wkey, minv)
        try:
            cur = int(cur) if cur not in (None, "") else minv
        except Exception:
            cur = minv

        disabled = False
        help_text = None
        if key == "idade":
            nascimento_wkey = KEY2WKEY.get("nascimento")
            birth_date = st.session_state.get(nascimento_wkey)
            if birth_date and isinstance(birth_date, date):
                calculated_age = calculate_age(birth_date)
                if calculated_age is not None:
                    cur = calculated_age
                    disabled = True
                    help_text = "Calculada automaticamente a partir da data de nascimento"

        st.number_input(label, min_value=minv, max_value=maxv, value=cur, key=wkey, disabled=disabled, help=help_text)

    elif ftype == "date":
        min_date = date(1900, 1, 1)
        max_date = date.today()

        if key == "nascimento":
            default_value = date(2000, 1, 1)
        elif key == "data":
            default_value = date.today()
        else:
            default_value = date.today()

        cur = st.session_state.get(wkey, default_value)
        if not isinstance(cur, date):
            cur = default_value

        if key == "nascimento":
            st.date_input(label, value=cur, min_value=min_date, max_value=max_date, key=wkey, on_change=update_age_from_birthdate)
        else:
            st.date_input(label, value=cur, min_value=min_date, max_value=max_date, key=wkey)

    elif ftype == "bool":
        st.checkbox(label, key=wkey)

    elif ftype == "yn":
        cur = st.session_state.get(wkey, "Não")
        idx = 0 if cur == "Não" else 1
        st.radio(label, options=["Não", "Sim"], horizontal=True, index=idx, key=wkey)

    elif ftype == "select":
        options = f.get("options", [""])
        if options and options[0] != "":
            options = [""] + options
        cur = st.session_state.get(wkey, options[0] if options else "")
        if options and cur not in options:
            cur = options[0]
        idx = options.index(cur) if options else 0
        st.selectbox(label, options=options, index=idx, key=wkey)

    elif ftype == "multiselect":
        options = f.get("options", [])
        cur = st.session_state.get(wkey, [])
        if not isinstance(cur, list):
            cur = []
        st.multiselect(label, options=options, default=cur, key=wkey)

    else:
        st.warning(f"Unsupported field type: {ftype} (key={key})")

def get_required_fields():
    return [f["key"] for f in FLAT_FIELDS if f.get("required", False)]

def validate_required_fields(row):
    required = get_required_fields()
    missing = []
    for field_key in required:
        value = row.get(field_key)
        if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
            field = next((f for f in FLAT_FIELDS if f["key"] == field_key), None)
            label = field["label"] if field else field_key
            missing.append(label)
    return missing

def build_row_from_state():
    values = {}
    for f in FLAT_FIELDS:
        k = f["key"]
        wkey = f["_widget_key"]
        v = st.session_state.get(wkey, None)

        if f.get("type") == "multiselect" and isinstance(v, list):
            v = json.dumps(v, ensure_ascii=False)

        values[k] = v

    row = {col: None for col in CSV_COLUMNS}
    for f in FLAT_FIELDS:
        k = f["key"]
        v = values.get(k, None)
        if "show_if" in f:
            dep_key, dep_val = f["show_if"]
            row[k] = v if values.get(dep_key) == dep_val else None
        else:
            row[k] = v
    return row

def save_row_to_csv(row: dict, csv_path: Path):
    df_new = pd.DataFrame([row], columns=CSV_COLUMNS)

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        for c in CSV_COLUMNS:
            if c not in df_old.columns:
                df_old[c] = pd.NA
        df_old = df_old[CSV_COLUMNS]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(csv_path, index=False, encoding="utf-8")

def load_csv_or_empty():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        for c in CSV_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[CSV_COLUMNS]
    return pd.DataFrame(columns=CSV_COLUMNS)

# --------------------------
# 5) Safe editable table (save only on button)
# --------------------------
YN_VALUES = {"Sim", "Não"}

def build_allowed_values(fields):
    allowed, types, mins, maxs = {}, {}, {}, {}
    for f in fields:
        k = f["key"]
        t = f.get("type", "text")
        if t == "options":
            t = "select"
        types[k] = t
        if t in ("select", "options"):
            allowed[k] = set(f.get("options", []))
        elif t == "yn":
            allowed[k] = set(YN_VALUES)
        else:
            allowed[k] = None
        if t == "number":
            mins[k] = f.get("min", 0)
            maxs[k] = f.get("max", 999)
    return allowed, types, mins, maxs

ALLOWED, TYPES, MINV, MAXV = build_allowed_values(FLAT_FIELDS)

def normalize_date_value(x):
    if pd.isna(x) or x in ("", None):
        return pd.NA
    if isinstance(x, (date, pd.Timestamp)):
        try:
            return pd.to_datetime(x).date().isoformat()
        except Exception:
            return pd.NA
    try:
        dt = pd.to_datetime(str(x), errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return pd.NA
        return dt.date().isoformat()
    except Exception:
        return pd.NA

def normalize_multiselect_value(x):
    if pd.isna(x) or x in ("", None):
        return pd.NA
    if isinstance(x, list):
        return json.dumps(x, ensure_ascii=False)
    s = str(x).strip()
    if not s:
        return pd.NA
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False)
    except Exception:
        pass
    return json.dumps([s], ensure_ascii=False)

def coerce_and_validate_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[CSV_COLUMNS].copy()

    for col in CSV_COLUMNS:
        t = TYPES.get(col, "text")

        if t == "number":
            s = pd.to_numeric(df[col], errors="coerce")
            minv = MINV.get(col, None)
            maxv = MAXV.get(col, None)
            if minv is not None:
                s = s.where((s >= minv) | s.isna(), pd.NA)
            if maxv is not None:
                s = s.where((s <= maxv) | s.isna(), pd.NA)
            df[col] = s

        elif t == "date":
            df[col] = df[col].apply(normalize_date_value)

        elif t == "yn":
            s = df[col].astype("string").str.strip()
            df[col] = s.where(s.isin(list(YN_VALUES)), pd.NA)

        elif t == "select":
            allowed = ALLOWED.get(col)
            s = df[col].astype("string").str.strip()
            if allowed and len(allowed) > 0:
                df[col] = s.where(s.isin(list(allowed)), pd.NA)
            else:
                df[col] = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

        elif t == "multiselect":
            df[col] = df[col].apply(normalize_multiselect_value)

        else:
            s = df[col].astype("string")
            df[col] = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return df

def save_df_to_csv_safe(df: pd.DataFrame, csv_path: Path):
    clean = coerce_and_validate_df(df)
    clean.to_csv(csv_path, index=False, encoding="utf-8")


# --------------------------
# 6) Analytics helpers
# --------------------------
def parse_iso_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt

def parse_multiselect_series(s: pd.Series) -> pd.Series:
    def _to_list(x):
        if pd.isna(x) or x in ("", None):
            return []
        if isinstance(x, list):
            return x
        try:
            v = json.loads(str(x))
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return s.apply(_to_list)

def make_age_group(age):
    try:
        if pd.isna(age):
            return pd.NA
        a = int(age)
    except Exception:
        return pd.NA
    bins = [(0, 17, "0–17"), (18, 24, "18–24"), (25, 34, "25–34"), (35, 44, "35–44"),
            (45, 54, "45–54"), (55, 64, "55–64"), (65, 200, "65+")]
    for lo, hi, lab in bins:
        if lo <= a <= hi:
            return lab
    return pd.NA

def kpi_cards(df_f: pd.DataFrame):
    total = len(df_f)
    # month + last 90 days + weekly avg (last 8 weeks)
    today = pd.Timestamp.today().normalize()
    month_start = today.replace(day=1)
    last_90 = today - pd.Timedelta(days=90)
    last_56 = today - pd.Timedelta(days=56)

    m = df_f[df_f["data_dt"] >= month_start].shape[0]
    q = df_f[df_f["data_dt"] >= last_90].shape[0]

    recent = df_f[df_f["data_dt"] >= last_56].copy()
    weekly_avg = None
    if not recent.empty:
        weekly = recent.set_index("data_dt").resample("W").size()
        weekly_avg = float(weekly.mean()) if len(weekly) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avaliações (total)", f"{total}")
    c2.metric("Avaliações (mês)", f"{m}")
    c3.metric("Avaliações (últimos 90 dias)", f"{q}")
    c4.metric("Média semanal (~8 semanas)", f"{weekly_avg:.1f}" if weekly_avg is not None else "—")

def bar_counts(df_f: pd.DataFrame, col: str, title: str):
    if col not in df_f.columns:
        return
    s = df_f[col].dropna().astype(str).replace({"": pd.NA}).dropna()
    if s.empty:
        st.info(f"Sem dados para: {title}")
        return
    vc = s.value_counts().reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(vc, x=col, y="count", title=title)
    st.plotly_chart(fig, use_container_width=True)

def hist_age(df_f: pd.DataFrame):
    if "idade" not in df_f.columns:
        return
    s = pd.to_numeric(df_f["idade"], errors="coerce").dropna()
    if s.empty:
        st.info("Sem dados de idade.")
        return
    p25, p50, p75 = s.quantile([0.25, 0.5, 0.75]).tolist()
    st.caption(f"Percentis idade: p25={p25:.0f}, p50={p50:.0f}, p75={p75:.0f}")
    fig = px.histogram(s, nbins=20, title="Distribuição de idade")
    st.plotly_chart(fig, use_container_width=True)

def time_series_counts(df_f: pd.DataFrame):
    if df_f.empty:
        return
    weekly = df_f.set_index("data_dt").resample("W").size().reset_index(name="count")
    weekly["Semana"] = weekly["data_dt"].dt.strftime("%Y-%m-%d")
    fig = px.line(weekly, x="Semana", y="count", markers=True, title="Avaliações por semana")
    st.plotly_chart(fig, use_container_width=True)

def spf_blocks(df_f: pd.DataFrame):
    # protetor_solar (Sim/Não) + dentro de Sim, freq stats
    if "protetor_solar" not in df_f.columns:
        return
    bar_counts(df_f, "protetor_solar", "Uso de protetor solar (Sim/Não)")

    if "protetor_solar_frequencia" in df_f.columns:
        d = df_f.copy()
        d["spf_freq"] = pd.to_numeric(d["protetor_solar_frequencia"], errors="coerce")
        d_sim = d[d["protetor_solar"] == "Sim"]["spf_freq"].dropna()
        if not d_sim.empty:
            st.caption(f"Entre quem usa SPF: média={d_sim.mean():.1f} | mediana={d_sim.median():.1f} (dias/semana)")
            fig = px.histogram(d_sim, nbins=8, title="Frequência de uso de protetor solar (dias/semana) — apenas 'Sim'")
            st.plotly_chart(fig, use_container_width=True)

def sun_blocks(df_f: pd.DataFrame):
    if "expor_ao_sol" in df_f.columns:
        bar_counts(df_f, "expor_ao_sol", "Exposição ao sol (Sim/Não)")
    if "expor_ao_sol_frequencia" in df_f.columns:
        s = pd.to_numeric(df_f["expor_ao_sol_frequencia"], errors="coerce").dropna()
        if not s.empty:
            fig = px.histogram(s, nbins=8, title="Frequência de exposição ao sol (dias/semana)")
            st.plotly_chart(fig, use_container_width=True)

def activity_blocks(df_f: pd.DataFrame):
    if "atividade_fisica" in df_f.columns:
        bar_counts(df_f, "atividade_fisica", "Atividade física (Sim/Não)")
    if "frequencia_atividade_fisica" in df_f.columns:
        s = pd.to_numeric(df_f["frequencia_atividade_fisica"], errors="coerce").dropna()
        if not s.empty:
            fig = px.histogram(s, nbins=8, title="Frequência de atividade física (dias/semana)")
            st.plotly_chart(fig, use_container_width=True)

def headache_blocks(df_f: pd.DataFrame):
    if "dores_de_cabeca" in df_f.columns:
        bar_counts(df_f, "dores_de_cabeca", "Dores de cabeça (Sim/Não)")
    if "frequencia_dores_cabeca" in df_f.columns:
        s = pd.to_numeric(df_f["frequencia_dores_cabeca"], errors="coerce").dropna()
        if not s.empty:
            fig = px.histogram(s, nbins=8, title="Frequência de dores de cabeça (dias/semana)")
            st.plotly_chart(fig, use_container_width=True)

def pigmentation_counts(df_f: pd.DataFrame):
    if "pigmentacao" not in df_f.columns:
        return
    lists = parse_multiselect_series(df_f["pigmentacao"])
    # explode to counts
    exploded = lists.explode()
    exploded = exploded[exploded.notna() & (exploded.astype(str).str.strip() != "")]
    if exploded.empty:
        st.info("Sem dados de pigmentação.")
        return
    vc = exploded.value_counts().reset_index()
    vc.columns = ["pigmentacao_opcao", "count"]
    fig = px.bar(vc, x="pigmentacao_opcao", y="count", title="Pigmentação — contagem por categoria (multiselect)")
    st.plotly_chart(fig, use_container_width=True)

def safety_flags_top(df_f: pd.DataFrame, topn=10):
    # pick YN fields that look like safety/medical flags
    yn_cols = [c for c in df_f.columns if TYPES.get(c) == "yn"]
    if not yn_cols:
        return

    # prevalence of "Sim"
    rows = []
    n = len(df_f)
    if n == 0:
        return
    for c in yn_cols:
        sim = (df_f[c] == "Sim").sum()
        if sim > 0:
            rows.append((c, sim, sim / n))
    if not rows:
        st.info("Sem flags 'Sim' registradas (ou sem dados).")
        return
    out = pd.DataFrame(rows, columns=["condicao", "count_sim", "pct_sim"]).sort_values("pct_sim", ascending=False).head(topn)
    out["pct_sim"] = (out["pct_sim"] * 100).round(1)

    fig = px.bar(out, x="condicao", y="pct_sim", title=f"Segurança/Elegibilidade — Top {topn} condições por prevalência (%)")
    fig.update_layout(xaxis_title="", yaxis_title="% com 'Sim'")
    st.plotly_chart(fig, use_container_width=True)

def relation_age_vs(df_f: pd.DataFrame):
    # idade × rugas/flacidez_grau
    if "idade" not in df_f.columns:
        return
    d = df_f.copy()
    d["idade_num"] = pd.to_numeric(d["idade"], errors="coerce")
    d["idade_grupo"] = d["idade_num"].apply(make_age_group)

    if "rugas" in d.columns:
        tmp = d.dropna(subset=["idade_grupo", "rugas"])
        tmp = tmp[tmp["rugas"].astype(str).str.strip() != ""]
        if not tmp.empty:
            ct = pd.crosstab(tmp["idade_grupo"], tmp["rugas"], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars="idade_grupo", var_name="rugas", value_name="pct")
            fig = px.bar(ct, x="idade_grupo", y="pct", color="rugas", barmode="stack",
                         title="Idade (grupos) × Rugas (distribuição % dentro de cada grupo)")
            st.plotly_chart(fig, use_container_width=True)

    if "flacidez_grau" in d.columns:
        tmp = d.dropna(subset=["idade_grupo", "flacidez_grau"])
        tmp = tmp[tmp["flacidez_grau"].astype(str).str.strip() != ""]
        if not tmp.empty:
            ct = pd.crosstab(tmp["idade_grupo"], tmp["flacidez_grau"], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars="idade_grupo", var_name="flacidez_grau", value_name="pct")
            fig = px.bar(ct, x="idade_grupo", y="pct", color="flacidez_grau", barmode="stack",
                         title="Idade (grupos) × Flacidez (distribuição % dentro de cada grupo)")
            st.plotly_chart(fig, use_container_width=True)

def relation_fototipo_pigment(df_f: pd.DataFrame):
    if "fototipo" not in df_f.columns or "pigmentacao" not in df_f.columns:
        return
    d = df_f.copy()
    d["pig_list"] = parse_multiselect_series(d["pigmentacao"])
    d = d.explode("pig_list")
    d["pig_list"] = d["pig_list"].astype(str).str.strip()
    d = d[(d["pig_list"] != "") & (~d["pig_list"].isna())]
    d = d[(d["fototipo"].astype(str).str.strip() != "") & (~d["fototipo"].isna())]
    if d.empty:
        st.info("Sem dados suficientes para fototipo × pigmentação.")
        return
    ct = pd.crosstab(d["fototipo"], d["pig_list"], normalize="index") * 100
    ct = ct.reset_index().melt(id_vars="fototipo", var_name="pigmentacao", value_name="pct")
    fig = px.bar(ct, x="fototipo", y="pct", color="pigmentacao", barmode="stack",
                 title="Fototipo × Pigmentação (distribuição % dentro de cada fototipo)")
    st.plotly_chart(fig, use_container_width=True)

def relation_sun_spf(df_f: pd.DataFrame):
    # expor_ao_sol_frequencia × protetor_solar_frequencia (scatter)
    if "expor_ao_sol_frequencia" not in df_f.columns or "protetor_solar_frequencia" not in df_f.columns:
        return
    d = df_f.copy()
    d["sun"] = pd.to_numeric(d["expor_ao_sol_frequencia"], errors="coerce")
    d["spf"] = pd.to_numeric(d["protetor_solar_frequencia"], errors="coerce")
    d = d.dropna(subset=["sun", "spf"])
    if d.empty:
        st.info("Sem dados suficientes para sol × SPF.")
        return
    fig = px.scatter(d, x="sun", y="spf", title="Frequência de sol × Frequência de SPF (dias/semana)",
                     labels={"sun": "Exposição ao sol (dias/semana)", "spf": "Uso de SPF (dias/semana)"})
    st.plotly_chart(fig, use_container_width=True)

def relation_smoker_skin(df_f: pd.DataFrame):
    if "fumante" not in df_f.columns:
        return
    if "hidratacao" in df_f.columns:
        tmp = df_f.dropna(subset=["fumante", "hidratacao"])
        tmp = tmp[(tmp["fumante"].astype(str).str.strip() != "") & (tmp["hidratacao"].astype(str).str.strip() != "")]
        if not tmp.empty:
            ct = pd.crosstab(tmp["fumante"], tmp["hidratacao"], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars="fumante", var_name="hidratacao", value_name="pct")
            fig = px.bar(ct, x="fumante", y="pct", color="hidratacao", barmode="stack",
                         title="Fumante × Hidratação (distribuição % dentro de fumante Sim/Não)")
            st.plotly_chart(fig, use_container_width=True)
    if "biotipo" in df_f.columns:
        tmp = df_f.dropna(subset=["fumante", "biotipo"])
        tmp = tmp[(tmp["fumante"].astype(str).str.strip() != "") & (tmp["biotipo"].astype(str).str.strip() != "")]
        if not tmp.empty:
            ct = pd.crosstab(tmp["fumante"], tmp["biotipo"], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars="fumante", var_name="biotipo", value_name="pct")
            fig = px.bar(ct, x="fumante", y="pct", color="biotipo", barmode="stack",
                         title="Fumante × Biotipo (distribuição % dentro de fumante Sim/Não)")
            st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 7) App UI (tabs)
# --------------------------
st.set_page_config(page_title="Ficha de Anamnese – Avaliação Facial", layout="wide")
st.title("Ficha de Anamnese – Avaliação Facial")

# Validation warnings
problems = validate_form_fields(FLAT_FIELDS)
if problems:
    with st.expander("⚠️ Dictionary validation warnings (click to see)", expanded=False):
        for p in problems:
            st.warning(p)

# Sidebar: dashboard access (owner-only)
with st.sidebar:
    st.subheader("🔒 Acesso Dashboard (dono/equipe)")
    st.caption("Configure a senha em st.secrets['DASHBOARD_PASS'] ou variável de ambiente DASHBOARD_PASS.")
    dash_pass_input = st.text_input("Senha", type="password")
    dash_pass = None
    try:
        dash_pass = st.secrets.get("DASHBOARD_PASS", None)
    except Exception:
        dash_pass = None
    if dash_pass is None:
        dash_pass = os.getenv("DASHBOARD_PASS", None)

    dashboard_ok = (dash_pass is not None) and (dash_pass_input == dash_pass)
    if dash_pass is None:
        st.warning("Senha do dashboard não configurada. (DASHBOARD_PASS)")
    elif not dashboard_ok:
        st.info("Digite a senha correta para ver os gráficos.")

tab_form, tab_dash, tab_csv = st.tabs(["📝 Formulário", "📊 Dashboard", "📄 CSV (tabela)"])

# -------- TAB 1: FORM --------
with tab_form:
    st.info("📝 Campos marcados com * são obrigatórios")

    for section, fields in FORM.items():
        with st.expander(section, expanded=True):
            for f in fields:
                f2 = dict(f)
                f2["_section"] = section
                f2["_widget_key"] = f"{section}__{f['key']}"
                render_field(f2)

    if st.button("Salvar", type="primary"):
        row = build_row_from_state()
        missing = validate_required_fields(row)

        if missing:
            st.error(f"⚠️ Por favor, preencha os seguintes campos obrigatórios: {', '.join(missing)}")
        else:
            save_row_to_csv(row, CSV_PATH)
            st.success("✅ Dados salvos com sucesso!")
            st.info(f"📁 Arquivo: {CSV_PATH.resolve()}")

# -------- TAB 2: DASHBOARD --------
with tab_dash:
    if not dashboard_ok:
        st.warning("Dashboard oculto. (Acesso restrito ao dono/equipe)")
    else:
        df_csv = load_csv_or_empty()
        if df_csv.empty:
            st.info("Ainda não há registros. Preencha o formulário e clique em 'Salvar'.")
        else:
            df = coerce_and_validate_df(df_csv)

            # parse dates + numeric
            df["data_dt"] = parse_iso_date_series(df["data"])
            df["idade_num"] = pd.to_numeric(df.get("idade", pd.Series(dtype="float")), errors="coerce")

            # Filters (per doc: data, idade grupo, sexo, biotipo, fototipo, hábitos chave) :contentReference[oaicite:4]{index=4}
            st.subheader("Filtros")

            c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1, 1, 1])
            min_d = df["data_dt"].min()
            max_d = df["data_dt"].max()
            if pd.isna(min_d) or pd.isna(max_d):
                st.warning("Datas inválidas na coluna 'data'. Verifique o CSV.")
                df_f = df.copy()
            else:
                with c1:
                    dr = st.date_input("Rango de datas (data da avaliação)", value=(min_d.date(), max_d.date()))
                start_d, end_d = dr
                start_dt = pd.to_datetime(start_d)
                end_dt = pd.to_datetime(end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                df_f = df[(df["data_dt"] >= start_dt) & (df["data_dt"] <= end_dt)].copy()

            df_f["idade_grupo"] = df_f["idade_num"].apply(make_age_group)

            with c2:
                age_opts = ["(todas)"] + [x for x in df_f["idade_grupo"].dropna().unique().tolist()]
                age_sel = st.selectbox("Grupo de idade", options=age_opts, index=0)
            if age_sel != "(todas)":
                df_f = df_f[df_f["idade_grupo"] == age_sel]

            with c3:
                sexo_opts = ["(todas)"] + [x for x in df_f["sexo"].dropna().astype(str).unique().tolist() if x.strip() != ""]
                sexo_sel = st.selectbox("Sexo", options=sexo_opts, index=0)
            if sexo_sel != "(todas)":
                df_f = df_f[df_f["sexo"] == sexo_sel]

            with c4:
                biotipo_opts = ["(todas)"] + [x for x in df_f["biotipo"].dropna().astype(str).unique().tolist() if x.strip() != ""]
                biotipo_sel = st.selectbox("Biotipo", options=biotipo_opts, index=0)
            if biotipo_sel != "(todas)":
                df_f = df_f[df_f["biotipo"] == biotipo_sel]

            with c5:
                fototipo_opts = ["(todas)"] + [x for x in df_f["fototipo"].dropna().astype(str).unique().tolist() if x.strip() != ""]
                fototipo_sel = st.selectbox("Fototipo", options=fototipo_opts, index=0)
            if fototipo_sel != "(todas)":
                df_f = df_f[df_f["fototipo"] == fototipo_sel]

            st.divider()
            st.subheader("Panorama (KPIs + Captação)")
            kpi_cards(df_f)  # :contentReference[oaicite:5]{index=5}
            time_series_counts(df_f)  # :contentReference[oaicite:6]{index=6}

            st.divider()
            st.subheader("Perfil demográfico")
            cA, cB = st.columns(2)
            with cA:
                hist_age(df_f)  # :contentReference[oaicite:7]{index=7}
            with cB:
                bar_counts(df_f, "sexo", "Distribuição por sexo")  # :contentReference[oaicite:8]{index=8}
                bar_counts(df_f, "idade_grupo", "Grupos de idade (derivado)")  # :contentReference[oaicite:9]{index=9}

            st.divider()
            st.subheader("Hábitos e autocuidado")
            cH1, cH2 = st.columns(2)
            with cH1:
                spf_blocks(df_f)  # :contentReference[oaicite:10]{index=10}
                bar_counts(df_f, "agua", "Ingestão diária de água")
                bar_counts(df_f, "sono", "Qualidade do sono")
            with cH2:
                sun_blocks(df_f)  # :contentReference[oaicite:11]{index=11}
                activity_blocks(df_f)  # :contentReference[oaicite:12]{index=12}
                headache_blocks(df_f)

            st.divider()
            st.subheader("Exame físico")
            cE1, cE2 = st.columns(2)
            with cE1:
                bar_counts(df_f, "biotipo", "Biotipo cutâneo")  # :contentReference[oaicite:13]{index=13}
                bar_counts(df_f, "hidratacao", "Grau de hidratação")  # :contentReference[oaicite:14]{index=14}
                bar_counts(df_f, "rugas", "Rugas")  # :contentReference[oaicite:15]{index=15}
                bar_counts(df_f, "flacidez_grau", "Flacidez")  # :contentReference[oaicite:16]{index=16}
            with cE2:
                bar_counts(df_f, "fototipo", "Fototipo (I–V)")  # :contentReference[oaicite:17]{index=17}
                pigmentation_counts(df_f)  # :contentReference[oaicite:18]{index=18}

            st.divider()
            st.subheader("Segurança / elegibilidade (flags)")
            safety_flags_top(df_f, topn=10)  # :contentReference[oaicite:19]{index=19}

            st.divider()
            st.subheader("Relações (segmentação / cruzamentos)")
            relation_age_vs(df_f)  # :contentReference[oaicite:20]{index=20}
            relation_fototipo_pigment(df_f)  # :contentReference[oaicite:21]{index=21}
            relation_sun_spf(df_f)  # :contentReference[oaicite:22]{index=22}
            relation_smoker_skin(df_f)  # :contentReference[oaicite:23]{index=23}

# -------- TAB 3: CSV (safe editor) --------
with tab_csv:
    if not dashboard_ok:
        st.warning("CSV oculto. (Acesso restrito ao dono/equipe)")
    else:
        st.subheader("📄 CSV Preview (Editable, Safe)")

        df_csv = load_csv_or_empty()
        if df_csv.empty:
            st.info("Nenhum dado registrado ainda. Preencha o formulário e clique em 'Salvar' para criar o arquivo CSV.")
        else:
            edited = st.data_editor(
                df_csv,
                use_container_width=True,
                num_rows="dynamic",
                key="csv_editor_safe",
            )

            clean_preview = coerce_and_validate_df(edited)

            colA, colB = st.columns([1, 1])
            with colA:
                if st.button("Salvar edições da tabela (seguro)"):
                    save_df_to_csv_safe(edited, CSV_PATH)
                    st.success("✅ Edições da tabela salvas com sucesso!")
                    st.info(f"📁 Arquivo: {CSV_PATH.resolve()}")

            with colB:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=clean_preview.to_csv(index=False).encode("utf-8"),
                    file_name=CSV_PATH.name,
                    mime="text/csv",
                )

