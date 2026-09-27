# -*- coding: utf-8 -*-
"""
=======================================================================
 AgroSearch - Motor de Busca Inteligente
 AgroTech Solutions | Recuperação da Informação
=======================================================================
 Protótipo de motor de busca textual construído INTEGRALMENTE do zero.

 RESTRIÇÃO ATENDIDA: nenhuma biblioteca de alto nível de NLP/ML é
 utilizada. Não há scikit-learn, NLTK, spaCy, gensim ou TfidfVectorizer.
 Tokenização, normalização, stopwords, stemming, índice invertido,
 TF, IDF, TF-IDF e similaridade de cosseno são implementados na mão,
 usando apenas a biblioteca padrão do Python (re, math, unicodedata,
 collections). Pandas é usado APENAS para renderizar tabelas na tela.

 Execução:  streamlit run agrosearch.py
=======================================================================
"""

import inspect
import math
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st

# =====================================================================
# 0. CONFIGURAÇÃO DA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="AgroSearch | Motor de Busca Inteligente",
    page_icon="🌱",
    layout="wide",
)

# ---------------------------------------------------------------------
# Camada de compatibilidade entre versões do Streamlit.
# Versões >= 1.49 usam width="stretch"; versões antigas usam
# use_container_width=True. Detectamos em tempo de execução para que o
# app rode em qualquer ambiente sem warnings de depreciação.
# ---------------------------------------------------------------------
def _largura_total(componente) -> dict:
    try:
        parametro = inspect.signature(componente).parameters.get("width")
    except (TypeError, ValueError):
        parametro = None
    if parametro is not None and isinstance(parametro.default, str):
        return {"width": "stretch"}
    return {"use_container_width": True}


LARGURA_TABELA = _largura_total(st.dataframe)
LARGURA_BOTAO = _largura_total(st.button)

# =====================================================================
# 1. BASE DE DOCUMENTOS (hardcode sugerido no enunciado)
# =====================================================================
CORPUS_PADRAO = [
    "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
]

# =====================================================================
# 2. FASE 1 - PIPELINE DE PRÉ-PROCESSAMENTO
# =====================================================================

# ---------------------------------------------------------------------
# 2.1 Lista de stopwords do português (construída manualmente).
# ---------------------------------------------------------------------
STOPWORDS_PT_BR = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as",
    "até", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "foram", "há", "isso", "isto", "já", "lhe", "lhes",
    "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito",
    "na", "nas", "nem", "no", "nos", "nós", "nossa", "nossas", "nosso",
    "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo",
    "pelos", "por", "qual", "quando", "que", "quem", "são", "se", "sem", "ser",
    "seu", "seus", "só", "sua", "suas", "também", "te", "tem", "tém", "tu",
    "um", "uma", "umas", "uns", "você", "vocês", "à", "às", "é", "ele",
    "esses", "pode", "ainda", "onde", "sobre", "todo", "toda", "todos",
    "todas", "outro", "outra", "cada", "seja", "sido", "tinha", "está",
    "estão", "havia", "isso", "aqui", "assim", "então", "mesma", "qualquer",
}


def normalizar(texto: str) -> str:
    """
    ETAPA A - NORMALIZAÇÃO.

    1) Converte para minúsculas (case folding);
    2) Remove acentos via decomposição Unicode NFD, descartando os
       caracteres da categoria 'Mn' (marcas de acentuação);
    3) Substitui qualquer caractere que não seja letra/dígito por espaço,
       eliminando pontuação e símbolos.

    Ex.: "Irrigação, ÁGUA!" -> "irrigacao  agua "
    """
    texto = texto.lower()
    decomposto = unicodedata.normalize("NFD", texto)
    sem_acento = "".join(c for c in decomposto if unicodedata.category(c) != "Mn")
    sem_acento = unicodedata.normalize("NFC", sem_acento)
    return re.sub(r"[^a-z0-9\s]", " ", sem_acento)


def tokenizar(texto_normalizado: str) -> list:
    """
    ETAPA B - TOKENIZAÇÃO.
    Quebra o texto em unidades léxicas (tokens) usando expressão regular.
    Tokens com 1 caractere são descartados (ruído: 'a', 'e', 'o'...).
    """
    return [t for t in re.findall(r"[a-z0-9]+", texto_normalizado) if len(t) > 1]


# Stopwords também passam pela normalização para que a comparação
# ocorra no mesmo "espaço" dos tokens (sem acento, minúsculas).
STOPWORDS_NORMALIZADAS = {normalizar(p).strip() for p in STOPWORDS_PT_BR}


def remover_stopwords(tokens: list) -> list:
    """ETAPA C - REMOÇÃO DE STOPWORDS (palavras vazias, sem poder discriminativo)."""
    return [t for t in tokens if t not in STOPWORDS_NORMALIZADAS]


# ---------------------------------------------------------------------
# 2.2 ETAPA D - STEMMING
# ---------------------------------------------------------------------
# Stemmer sufixal para o português, inspirado no algoritmo RSLP
# (Removedor de Sufixos da Língua Portuguesa, Orengo & Huyck, 2001),
# porém implementado do zero e de forma simplificada/didática.
#
# As regras operam sobre texto JÁ NORMALIZADO (sem acentos). Por isso
# "irrigação" chega aqui como "irrigacao" e o sufixo tratado é "acao".
#
# Cada regra é uma tupla: (sufixo, tamanho_mínimo_do_radical, substituto)
# A regra só é aplicada se o radical resultante tiver tamanho suficiente,
# o que evita mutilar palavras curtas.
# ---------------------------------------------------------------------

REGRAS_PLURAL = [
    ("ns", 1, "m"), ("oes", 3, "ao"), ("aes", 1, "ao"), ("ais", 1, "al"),
    ("eis", 2, "el"), ("ois", 2, "ol"), ("is", 2, "il"), ("les", 2, "l"),
    ("res", 3, "r"), ("s", 2, ""),
]

REGRAS_FEMININO = [
    ("inha", 3, "inho"), ("ona", 3, "ao"), ("ora", 3, "or"),
    ("eira", 3, "eiro"), ("osa", 3, "oso"), ("ica", 3, "ico"),
    ("ida", 3, "ido"), ("ada", 3, "ado"), ("iva", 3, "ivo"),
    ("ima", 3, "imo"), ("na", 4, "no"),
]

REGRAS_ADVERBIO = [("mente", 4, "")]

REGRAS_AUMENTATIVO_DIMINUTIVO = [
    ("issimo", 3, ""), ("zinho", 2, ""), ("zinha", 2, ""), ("inho", 3, ""),
    ("inha", 3, ""), ("zao", 2, ""), ("zona", 2, ""),
]

REGRAS_NOMINAL = [
    ("amento", 3, ""), ("imento", 3, ""), ("mento", 3, ""),
    ("acao", 3, ""), ("icao", 3, ""), ("cao", 3, ""), ("sao", 3, ""),
    ("idade", 3, ""), ("dade", 3, ""), ("ancia", 3, ""), ("encia", 3, ""),
    ("ista", 3, ""), ("ismo", 3, ""), ("ador", 3, ""), ("edor", 3, ""),
    ("idor", 3, ""), ("agem", 3, ""), ("ario", 3, ""), ("eiro", 3, ""),
    ("eira", 3, ""), ("oso", 3, ""), ("osa", 3, ""), ("ivo", 3, ""),
    ("iva", 3, ""), ("eza", 3, ""), ("ura", 3, ""), ("ico", 3, ""),
    ("ica", 3, ""), ("ncia", 3, ""),
]

REGRAS_VERBAL = [
    ("ariam", 2, ""), ("eriam", 2, ""), ("iriam", 2, ""),
    ("assem", 2, ""), ("essem", 2, ""), ("issem", 2, ""),
    ("aram", 2, ""), ("eram", 2, ""), ("iram", 2, ""),
    ("ando", 2, ""), ("endo", 2, ""), ("indo", 2, ""),
    ("aria", 2, ""), ("eria", 2, ""), ("iria", 2, ""),
    ("avam", 2, ""), ("ante", 2, ""), ("ente", 3, ""),
    ("adas", 2, ""), ("idas", 2, ""), ("ados", 2, ""), ("idos", 2, ""),
    ("ava", 2, ""), ("ado", 2, ""), ("ido", 2, ""),
    ("ir", 2, ""), ("er", 2, ""), ("ar", 2, ""),
    ("am", 2, ""), ("em", 2, ""), ("ou", 3, ""), ("ei", 3, ""),
]

VOGAIS_FINAIS = [("a", 3, ""), ("e", 3, ""), ("o", 3, "")]


def _aplicar_regras(palavra: str, regras: list):
    """
    Tenta aplicar a PRIMEIRA regra compatível da lista.
    Retorna a palavra modificada ou None se nenhuma regra se aplicar.
    """
    for sufixo, tam_min, substituto in regras:
        if palavra.endswith(sufixo):
            radical = palavra[: len(palavra) - len(sufixo)]
            if len(radical) >= tam_min:
                return radical + substituto
    return None


def stemizar(palavra: str) -> str:
    """
    Reduz a palavra ao seu radical (stem) aplicando os blocos de regras
    na ordem canônica do RSLP. O objetivo é fazer com que variações
    como "irrigação", "irrigar" e "irrigado" colapsem no mesmo termo
    de indexação ("irrig"), aumentando o recall da busca.
    """
    if len(palavra) <= 3:
        return palavra

    # 1) Plural (apenas se terminar em 's')
    if palavra.endswith("s"):
        nova = _aplicar_regras(palavra, REGRAS_PLURAL)
        if nova:
            palavra = nova

    # 2) Feminino (apenas se terminar em 'a')
    if palavra.endswith("a"):
        nova = _aplicar_regras(palavra, REGRAS_FEMININO)
        if nova:
            palavra = nova

    # 3) Advérbio
    nova = _aplicar_regras(palavra, REGRAS_ADVERBIO)
    if nova:
        palavra = nova

    # 4) Aumentativo / Diminutivo
    nova = _aplicar_regras(palavra, REGRAS_AUMENTATIVO_DIMINUTIVO)
    if nova:
        palavra = nova

    # 5) Sufixos nominais
    nova = _aplicar_regras(palavra, REGRAS_NOMINAL)
    if nova:
        palavra = nova
    else:
        # 6) Sufixos verbais (só se nenhum nominal foi removido)
        nova = _aplicar_regras(palavra, REGRAS_VERBAL)
        if nova:
            palavra = nova

    # 7) Remoção da vogal temática final
    nova = _aplicar_regras(palavra, VOGAIS_FINAIS)
    if nova:
        palavra = nova

    return palavra


def aplicar_stemming(tokens: list) -> list:
    return [stemizar(t) for t in tokens]


def preprocessar(texto: str, usar_stopwords: bool, usar_stemming: bool) -> dict:
    """
    PIPELINE COMPLETO. Devolve um dicionário com o resultado de CADA
    etapa, permitindo que a interface mostre a transformação passo a passo.
    """
    texto_normalizado = normalizar(texto)
    tokens = tokenizar(texto_normalizado)
    tokens_sem_sw = remover_stopwords(tokens) if usar_stopwords else list(tokens)
    tokens_finais = aplicar_stemming(tokens_sem_sw) if usar_stemming else list(tokens_sem_sw)

    return {
        "original": texto,
        "normalizado": texto_normalizado,
        "tokens": tokens,
        "sem_stopwords": tokens_sem_sw,
        "final": tokens_finais,
    }


# =====================================================================
# 2.5 LEITURA DE ARQUIVOS ENVIADOS (PDF / TXT)
# =====================================================================
# Observação sobre a restrição do desafio: ler um PDF é desempacotar um
# formato binário de arquivo, não processar linguagem natural. Nenhuma
# etapa de NLP/IR é terceirizada — a biblioteca apenas devolve a string
# bruta, que em seguida passa pelo NOSSO pipeline e pelo NOSSO TF-IDF.
# =====================================================================

def _localizar_leitor_pdf():
    """Procura uma biblioteca de leitura de PDF disponível no ambiente."""
    for nome_modulo in ("pypdf", "PyPDF2"):
        try:
            modulo = __import__(nome_modulo)
            return getattr(modulo, "PdfReader"), nome_modulo
        except (ImportError, AttributeError):
            continue
    return None, None


def extrair_texto_pdf(arquivo):
    """
    Extrai o texto de um PDF enviado pelo usuário.
    Retorna a tupla (texto, número_de_páginas, mensagem_de_erro).
    """
    PdfReader, _ = _localizar_leitor_pdf()
    if PdfReader is None:
        return "", 0, "Biblioteca ausente — instale com: pip install pypdf"
    try:
        leitor = PdfReader(arquivo)
        paginas = [(pagina.extract_text() or "") for pagina in leitor.pages]
    except Exception as erro:  # PDF corrompido, protegido por senha etc.
        return "", 0, f"Falha ao abrir o arquivo ({type(erro).__name__})."

    texto = re.sub(r"\s+", " ", " ".join(paginas)).strip()
    if not texto:
        return "", len(paginas), ("Nenhum texto extraível — provável PDF "
                                  "digitalizado, exigiria OCR.")
    return texto, len(paginas), ""


def extrair_texto_txt(arquivo):
    """Lê um arquivo de texto simples, tentando UTF-8 e depois Latin-1."""
    dados = arquivo.read()
    for codificacao in ("utf-8", "latin-1"):
        try:
            texto = re.sub(r"\s+", " ", dados.decode(codificacao)).strip()
            return texto, 1, "" if texto else "Arquivo vazio."
        except UnicodeDecodeError:
            continue
    return "", 0, "Não foi possível decodificar o arquivo."


def carregar_arquivos(arquivos):
    """
    Converte a lista de arquivos enviados em um corpus.
    Cada arquivo vira um documento, rotulado pelo próprio nome.
    Retorna (textos, relatorio_de_extracao).
    """
    textos, relatorio, rotulos_usados = {}, [], set()

    for arquivo in arquivos:
        nome = arquivo.name
        if nome.lower().endswith(".pdf"):
            texto, paginas, erro = extrair_texto_pdf(arquivo)
        else:
            texto, paginas, erro = extrair_texto_txt(arquivo)

        rotulo = re.sub(r"\.(pdf|txt)$", "", nome, flags=re.IGNORECASE)[:22]
        rotulo_base, sufixo = rotulo, 2
        while rotulo in rotulos_usados:  # garante rótulos únicos
            rotulo = f"{rotulo_base} ({sufixo})"
            sufixo += 1
        rotulos_usados.add(rotulo)

        relatorio.append({
            "Arquivo": nome,
            "Rótulo no corpus": rotulo,
            "Páginas": paginas,
            "Caracteres": len(texto),
            "Situação": "✅ texto extraído" if texto else f"⚠️ {erro}",
            "Prévia": (texto[:90] + "…") if len(texto) > 90 else texto,
        })
        if texto:
            textos[rotulo] = texto

    return textos, relatorio


# =====================================================================
# 3. FASE 2 - ÍNDICE INVERTIDO
# =====================================================================

def construir_indice_invertido(docs_tokens: dict) -> dict:
    """
    Constrói o índice invertido em memória:

        { termo : { doc_id : frequência_do_termo_no_doc } }

    Essa estrutura é o coração do motor de busca: em vez de varrer todos
    os documentos a cada consulta (busca linear O(N*M)), consultamos
    diretamente a "posting list" do termo em O(1) no dicionário.
    """
    indice = defaultdict(dict)
    for doc_id, tokens in docs_tokens.items():
        for termo, freq in Counter(tokens).items():
            indice[termo][doc_id] = freq
    return dict(sorted(indice.items()))


# =====================================================================
# 4. FASE 3 - TF, IDF, TF-IDF E RANQUEAMENTO
# =====================================================================

def calcular_tf(freq: int, total_tokens: int, esquema: str) -> float:
    """
    TF (Term Frequency) - o quanto o termo é importante DENTRO do documento.

      • Bruto        : tf = f(t,d)
      • Normalizado  : tf = f(t,d) / |d|          (corrige viés de tamanho)
      • Logarítmico  : tf = 1 + log10(f(t,d))     (satura ganhos repetidos)
    """
    if freq <= 0:
        return 0.0
    if esquema == "Bruto: f(t,d)":
        return float(freq)
    if esquema == "Normalizado: f(t,d) / |d|":
        return freq / total_tokens if total_tokens else 0.0
    return 1.0 + math.log10(freq)  # Logarítmico


def calcular_idf(n_docs: int, df: int, esquema: str) -> float:
    """
    IDF (Inverse Document Frequency) - o quanto o termo é RARO no corpus.
    Termos que aparecem em todos os documentos têm poder discriminativo
    baixo (ou nulo, no esquema clássico).

      • Clássico  : idf = log10(N / df)
      • Suavizado : idf = log10(N / df) + 1        (nunca zera)
      • Sklearn   : idf = log10((1+N)/(1+df)) + 1  (evita divisão por zero)
    """
    if df <= 0:
        return 0.0
    if esquema == "Clássico: log10(N/df)":
        return math.log10(n_docs / df)
    if esquema == "Suavizado: log10(N/df) + 1":
        return math.log10(n_docs / df) + 1.0
    return math.log10((1 + n_docs) / (1 + df)) + 1.0


def construir_matriz_tfidf(docs_tokens, indice, esquema_tf, esquema_idf):
    """
    Calcula a matriz TF-IDF completa: {doc_id: {termo: peso}}
    e o vetor de IDF de cada termo do vocabulário.

        tfidf(t,d) = tf(t,d) * idf(t)
    """
    n_docs = len(docs_tokens)
    tamanhos = {d: len(tks) for d, tks in docs_tokens.items()}

    idfs = {
        termo: calcular_idf(n_docs, len(postings), esquema_idf)
        for termo, postings in indice.items()
    }

    matriz = {doc_id: {} for doc_id in docs_tokens}
    for termo, postings in indice.items():
        for doc_id, freq in postings.items():
            tf = calcular_tf(freq, tamanhos[doc_id], esquema_tf)
            matriz[doc_id][termo] = tf * idfs[termo]

    return matriz, idfs


def buscar(query_tokens, docs_tokens, indice, matriz_tfidf, idfs,
           esquema_tf, esquema_idf):
    """
    RANQUEAMENTO.

    1) Score TF-IDF acumulado (soma dos pesos dos termos da query):
           score(q,d) = Σ_{t ∈ q}  tfidf(t,d)

    2) BÔNUS - Similaridade de Cosseno entre o vetor da query e o vetor
       do documento no espaço vetorial do vocabulário:

           cos(q,d) = (q · d) / (||q|| * ||d||)

       O cosseno normaliza o tamanho dos vetores, tratando melhor
       consultas com múltiplas palavras e documentos longos.
    """
    n_docs = len(docs_tokens)

    # --- Vetor da consulta: tf da query * idf do corpus -------------
    freq_query = Counter(query_tokens)
    vetor_query = {}
    for termo, freq in freq_query.items():
        idf = idfs.get(termo, calcular_idf(n_docs, 0, esquema_idf))
        tf = calcular_tf(freq, len(query_tokens), esquema_tf)
        vetor_query[termo] = tf * idf

    norma_query = math.sqrt(sum(v ** 2 for v in vetor_query.values()))

    resultados = []
    for doc_id, vetor_doc in matriz_tfidf.items():
        soma_tfidf = sum(vetor_doc.get(t, 0.0) for t in freq_query)
        termos_encontrados = [t for t in freq_query if t in vetor_doc]

        produto_escalar = sum(
            peso_q * vetor_doc.get(termo, 0.0)
            for termo, peso_q in vetor_query.items()
        )
        norma_doc = math.sqrt(sum(v ** 2 for v in vetor_doc.values()))
        cosseno = (produto_escalar / (norma_query * norma_doc)
                   if norma_query and norma_doc else 0.0)

        resultados.append({
            "doc_id": doc_id,
            "score_tfidf": soma_tfidf,
            "cosseno": cosseno,
            "termos_casados": termos_encontrados,
            "cobertura": len(termos_encontrados) / len(freq_query) if freq_query else 0.0,
        })

    return resultados, vetor_query


# =====================================================================
# 5. UTILITÁRIOS DE INTERFACE
# =====================================================================

def destacar_termos(texto_original: str, termos_alvo: set,
                    usar_stopwords: bool, usar_stemming: bool) -> str:
    """
    Destaca no texto ORIGINAL as palavras cujo token processado casa com
    algum termo da consulta. Demonstra visualmente o efeito do stemming
    (ex.: a query "irrigar" acende a palavra "irrigação" no documento).
    """
    if not termos_alvo:
        return texto_original

    def substituir(match):
        palavra = match.group(0)
        proc = preprocessar(palavra, usar_stopwords, usar_stemming)["final"]
        if proc and proc[0] in termos_alvo:
            return f"**:orange[{palavra}]**"
        return palavra

    return re.sub(r"\w+", substituir, texto_original, flags=re.UNICODE)


def barra_visual(valor: float, maximo: float, largura: int = 18) -> str:
    if maximo <= 0:
        return ""
    preenchido = int(round((valor / maximo) * largura))
    return "█" * preenchido + "░" * (largura - preenchido)


# =====================================================================
# 6. APLICAÇÃO STREAMLIT
# =====================================================================

def main():
    st.title("🌱 AgroSearch")
    st.caption(
        "Motor de busca textual **construído do zero** — pipeline de "
        "pré-processamento, índice invertido e ranqueamento TF-IDF. "
        "AgroTech Solutions."
    )

    # -----------------------------------------------------------------
    # SIDEBAR - CONTROLES
    # -----------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuração do Pipeline")

        usar_stopwords = st.checkbox(
            "Remover Stopwords", value=True,
            help="Descarta palavras funcionais sem poder discriminativo "
                 "(a, de, para, com...).",
        )
        usar_stemming = st.checkbox(
            "Aplicar Stemming", value=True,
            help="Reduz as palavras ao radical. 'irrigação', 'irrigar' e "
                 "'irrigado' viram o mesmo termo.",
        )

        st.divider()
        st.header("📐 Fórmulas de Ponderação")
        esquema_tf = st.selectbox(
            "Esquema de TF",
            ["Normalizado: f(t,d) / |d|", "Bruto: f(t,d)", "Logarítmico: 1 + log10(f)"],
        )
        esquema_idf = st.selectbox(
            "Esquema de IDF",
            ["Clássico: log10(N/df)", "Suavizado: log10(N/df) + 1",
             "Suavizado (+1): log10((1+N)/(1+df)) + 1"],
        )

        st.divider()
        st.header("📚 Base de Documentos")
        fonte = st.radio(
            "Fonte dos documentos:",
            ["Base padrão (texto)", "Upload de PDFs"],
            help="Envie os manuais técnicos em PDF: o sistema extrai o texto "
                 "e o submete ao mesmo pipeline dos documentos internos.",
        )

        arquivos = []
        texto_corpus = "\n".join(CORPUS_PADRAO)

        if fonte == "Upload de PDFs":
            arquivos = st.file_uploader(
                "Manuais técnicos (.pdf ou .txt)",
                type=["pdf", "txt"],
                accept_multiple_files=True,
            ) or []
            st.caption(
                f"{len(arquivos)} arquivo(s) enviado(s). Cada arquivo vira um "
                "documento do corpus, rotulado pelo nome."
            )
        else:
            with st.expander("Editar corpus (1 documento por linha)"):
                texto_corpus = st.text_area(
                    "Documentos", value="\n".join(CORPUS_PADRAO),
                    height=220, label_visibility="collapsed",
                    key="corpus_editado",
                )
                if st.button("↺ Restaurar base padrão", **LARGURA_BOTAO):
                    st.session_state.pop("corpus_editado", None)
                    st.rerun()

        st.divider()
        st.caption(
            "Sem scikit-learn, NLTK ou TfidfVectorizer. "
            "Apenas `re`, `math`, `unicodedata` e `collections`."
        )

    # -----------------------------------------------------------------
    # CARGA DO CORPUS
    # -----------------------------------------------------------------
    if fonte == "Upload de PDFs":
        if not arquivos:
            st.info(
                "⬅️ **Envie os manuais em PDF pela barra lateral** para que o "
                "AgroSearch extraia o texto, construa o índice invertido e "
                "responda às consultas. Você também pode voltar para a base "
                "padrão em texto."
            )
            st.stop()

        textos, relatorio_extracao = carregar_arquivos(arquivos)
        if not textos:
            st.error("Nenhum texto pôde ser extraído dos arquivos enviados.")
            st.dataframe(pd.DataFrame(relatorio_extracao), hide_index=True,
                         **LARGURA_TABELA)
            st.stop()

        doc_ids = list(textos.keys())
        documentos = list(textos.values())
    else:
        relatorio_extracao = []
        documentos = [linha.strip() for linha in texto_corpus.split("\n")
                      if linha.strip()]
        if not documentos:
            st.error("A base de documentos está vazia. Adicione ao menos um documento.")
            st.stop()
        doc_ids = [f"Doc {i + 1}" for i in range(len(documentos))]
        textos = dict(zip(doc_ids, documentos))

    # -----------------------------------------------------------------
    # EXECUÇÃO DO PIPELINE
    # -----------------------------------------------------------------
    processados = {
        doc_id: preprocessar(txt, usar_stopwords, usar_stemming)
        for doc_id, txt in textos.items()
    }
    docs_tokens = {doc_id: p["final"] for doc_id, p in processados.items()}

    indice = construir_indice_invertido(docs_tokens)
    matriz_tfidf, idfs = construir_matriz_tfidf(
        docs_tokens, indice, esquema_tf, esquema_idf
    )

    # -----------------------------------------------------------------
    # PAINEL DE MÉTRICAS
    # -----------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    total_tokens = sum(len(t) for t in docs_tokens.values())
    c1.metric("Documentos (N)", len(documentos))
    c2.metric("Tokens indexados", total_tokens)
    c3.metric("Vocabulário", len(indice))
    c4.metric(
        "Taxa de compressão",
        f"{(1 - len(indice) / total_tokens) * 100:.0f}%" if total_tokens else "—",
        help="Redução do vocabulário em relação ao total de tokens.",
    )

    aba1, aba2, aba3, aba4 = st.tabs([
        "1️⃣ Pré-processamento",
        "2️⃣ Índice Invertido",
        "3️⃣ Busca & Ranqueamento",
        "📖 Fundamentos",
    ])

    # =================================================================
    # ABA 1 - PRÉ-PROCESSAMENTO
    # =================================================================
    with aba1:
        st.subheader("Pipeline de Pré-processamento")

        if relatorio_extracao:
            st.markdown("##### 📄 Inspeção da extração dos arquivos")
            st.dataframe(pd.DataFrame(relatorio_extracao), hide_index=True,
                         **LARGURA_TABELA)
            falhas = [r for r in relatorio_extracao if "⚠️" in r["Situação"]]
            if falhas:
                st.warning(
                    f"{len(falhas)} arquivo(s) não entraram no corpus. PDFs "
                    "digitalizados (imagem) não têm camada de texto e exigiriam "
                    "OCR antes da indexação."
                )
            else:
                st.caption(
                    "Texto extraído de todos os arquivos. A partir daqui, o "
                    "conteúdo segue exatamente o mesmo pipeline dos documentos "
                    "internos."
                )
            st.divider()

        st.markdown(
            "Ligue e desligue **Stopwords** e **Stemming** na barra lateral "
            "e observe o vocabulário mudar dinamicamente."
        )

        estado = []
        if usar_stopwords:
            estado.append("Stopwords ✅")
        else:
            estado.append("Stopwords ❌")
        estado.append("Stemming ✅" if usar_stemming else "Stemming ❌")
        st.info(" • ".join(estado) + f" — vocabulário atual: **{len(indice)} termos**")

        doc_escolhido = st.selectbox("Inspecionar documento:", doc_ids)
        p = processados[doc_escolhido]

        st.markdown("##### Transformação passo a passo")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**0. Texto original**")
            st.code(p["original"], language=None)
            st.markdown("**1. Normalização** *(minúsculas, sem acento, sem pontuação)*")
            st.code(p["normalizado"].strip(), language=None)
        with e2:
            st.markdown(f"**2. Tokenização** — {len(p['tokens'])} tokens")
            st.code(", ".join(p["tokens"]), language=None)
            rot3 = "3. Remoção de Stopwords" if usar_stopwords else "3. Stopwords (desativado)"
            st.markdown(f"**{rot3}** — {len(p['sem_stopwords'])} tokens")
            st.code(", ".join(p["sem_stopwords"]), language=None)

        rot4 = "4. Stemming" if usar_stemming else "4. Stemming (desativado)"
        st.markdown(f"**{rot4}** — tokens finais que vão para o índice")
        st.code(", ".join(p["final"]), language=None)

        if usar_stemming:
            pares = [
                {"Token": a, "Stem": b, "Reduziu?": "sim" if a != b else "—"}
                for a, b in zip(p["sem_stopwords"], p["final"])
            ]
            with st.expander("🔍 Ver o efeito do stemmer token a token"):
                st.dataframe(pd.DataFrame(pares), hide_index=True,
                             **LARGURA_TABELA)

        st.divider()
        st.markdown("##### Vocabulário global do corpus")
        st.write(sorted(indice.keys()))

        with st.expander("🧪 Testar o pipeline em um texto qualquer"):
            teste = st.text_input("Digite uma frase:",
                                  "As lagartas desfolhadoras exigem irrigação!")
            if teste:
                pt = preprocessar(teste, usar_stopwords, usar_stemming)
                st.write("Normalizado:", f"`{pt['normalizado'].strip()}`")
                st.write("Tokens:", pt["tokens"])
                st.write("Final:", pt["final"])

    # =================================================================
    # ABA 2 - ÍNDICE INVERTIDO
    # =================================================================
    with aba2:
        st.subheader("Índice Invertido")
        st.markdown(
            "Estrutura `Termo → [IDs dos Documentos]`. É ela que evita a "
            "varredura linear do corpus: a consulta salta direto para a "
            "*posting list* do termo."
        )

        visao = st.radio("Formato de exibição:",
                         ["Tabela", "JSON (termo → docs)", "JSON (com frequências)"],
                         horizontal=True)

        if visao == "Tabela":
            filtro = st.text_input("Filtrar termo:", "")
            linhas = []
            for termo, postings in indice.items():
                if filtro and filtro.lower() not in termo:
                    continue
                linhas.append({
                    "Termo": termo,
                    "DF (nº de docs)": len(postings),
                    "Posting List": ", ".join(sorted(postings.keys())),
                    "Frequências": ", ".join(f"{d}:{f}" for d, f in sorted(postings.items())),
                    "IDF": round(idfs[termo], 4),
                })
            df_idx = pd.DataFrame(linhas)
            st.dataframe(df_idx, hide_index=True, **LARGURA_TABELA, height=420)
            st.caption(f"{len(linhas)} termo(s) exibido(s) de {len(indice)} no vocabulário.")
        elif visao == "JSON (termo → docs)":
            st.json({t: sorted(p.keys()) for t, p in indice.items()})
        else:
            st.json({t: p for t, p in indice.items()})

        st.divider()
        st.markdown("##### Matriz TF-IDF (documentos × termos)")
        st.caption(
            "Cada célula é o peso `tf(t,d) × idf(t)`. Colunas inteiras zeradas "
            "indicam termos presentes em TODOS os documentos (IDF = 0 no "
            "esquema clássico) — eles não discriminam nada."
        )
        df_matriz = pd.DataFrame(matriz_tfidf).T.fillna(0.0)
        df_matriz = df_matriz.reindex(sorted(df_matriz.columns), axis=1).round(4)
        try:
            # O mapa de calor exige matplotlib; se não houver, exibe sem cor.
            st.dataframe(
                df_matriz.style.background_gradient(cmap="Greens", axis=None),
                **LARGURA_TABELA,
            )
        except ImportError:
            st.dataframe(df_matriz, **LARGURA_TABELA)

    # =================================================================
    # ABA 3 - BUSCA E RANQUEAMENTO
    # =================================================================
    with aba3:
        st.subheader("Busca e Ranqueamento por Relevância")

        if "query" not in st.session_state:
            st.session_state["query"] = "irrigação da soja"

        st.caption("Sugestões rápidas:")
        sug = st.columns(4)
        for col, texto_sug in zip(sug, ["lagartas na soja", "irrigação",
                                        "solo e nitrogênio", "cultivo orgânico"]):
            if col.button(texto_sug, **LARGURA_BOTAO):
                st.session_state["query"] = texto_sug

        col_q, col_m = st.columns([3, 2])
        with col_q:
            query = st.text_input(
                "🔎 Consulta do técnico de campo:",
                key="query",
                placeholder="ex.: controle de lagartas",
            )
        with col_m:
            metrica = st.radio(
                "Ordenar por:",
                ["TF-IDF acumulado", "Similaridade de Cosseno (bônus)"],
                help="O cosseno normaliza o tamanho dos vetores e lida melhor "
                     "com consultas de múltiplas palavras.",
            )

        pq = preprocessar(query, usar_stopwords, usar_stemming)
        query_tokens = pq["final"]

        st.markdown(
            f"**Consulta processada:** `{query if query.strip() else '(vazia)'}` → "
            f"`{query_tokens if query_tokens else '∅'}`"
        )

        if not query_tokens:
            st.info(
                "Digite uma consulta com ao menos um termo de conteúdo. "
                "Consultas formadas apenas por stopwords (ex.: *\"de para o\"*) "
                "ficam vazias após o pré-processamento e não recuperam nada."
            )
        else:
            ausentes = [t for t in dict.fromkeys(query_tokens) if t not in indice]
            if ausentes:
                st.warning(
                    f"Termo(s) fora do vocabulário (df = 0, não recuperam nada): "
                    f"`{', '.join(ausentes)}`"
                )

            resultados, vetor_query = buscar(
                query_tokens, docs_tokens, indice, matriz_tfidf, idfs,
                esquema_tf, esquema_idf,
            )

            chave = "score_tfidf" if metrica.startswith("TF-IDF") else "cosseno"
            resultados.sort(key=lambda r: (r[chave], r["cobertura"]), reverse=True)

            # --- Tabela de ranqueamento -----------------------------
            maximo = max((r[chave] for r in resultados), default=0.0)
            linhas_rank = []
            for pos, r in enumerate(resultados, start=1):
                linhas_rank.append({
                    "#": pos,
                    "Documento": r["doc_id"],
                    "TF-IDF acumulado": round(r["score_tfidf"], 5),
                    "Cosseno": round(r["cosseno"], 5),
                    "Relevância": barra_visual(r[chave], maximo),
                    "Termos casados": ", ".join(r["termos_casados"]) or "—",
                    "Cobertura": f"{r['cobertura'] * 100:.0f}%",
                    "Trecho": textos[r["doc_id"]][:70] + "…",
                })

            st.markdown("##### 🏅 Ranking (maior → menor)")
            st.dataframe(pd.DataFrame(linhas_rank), hide_index=True,
                         **LARGURA_TABELA)

            # --- Documento vencedor ---------------------------------
            vencedor = resultados[0]
            if vencedor[chave] <= 0:
                st.error(
                    "Nenhum documento relevante encontrado para esta consulta. "
                    "Observação: se um termo aparece em TODOS os documentos, o "
                    "IDF clássico o zera — experimente um esquema suavizado na "
                    "barra lateral."
                )
            else:
                st.success(f"🏆 **Documento vencedor: {vencedor['doc_id']}**")
                v1, v2, v3 = st.columns(3)
                v1.metric("TF-IDF acumulado", f"{vencedor['score_tfidf']:.5f}")
                v2.metric("Similaridade de Cosseno", f"{vencedor['cosseno']:.5f}")
                v3.metric("Cobertura da query", f"{vencedor['cobertura'] * 100:.0f}%")

                st.markdown("**Trecho recuperado** *(termos casados destacados)*:")
                st.markdown(
                    "> " + destacar_termos(
                        textos[vencedor["doc_id"]], set(query_tokens),
                        usar_stopwords, usar_stemming,
                    )
                )

                if len(resultados) > 1 and resultados[1][chave] > 0:
                    margem = vencedor[chave] - resultados[1][chave]
                    st.caption(
                        f"Vantagem sobre {resultados[1]['doc_id']} "
                        f"(2º colocado): {margem:.5f}."
                    )

            # --- Memória de cálculo ---------------------------------
            st.divider()
            with st.expander("🧮 Memória de cálculo termo a termo", expanded=True):
                n_docs = len(documentos)
                st.markdown(
                    f"`TF: {esquema_tf}` &nbsp;|&nbsp; `IDF: {esquema_idf}` "
                    f"&nbsp;|&nbsp; `N = {n_docs}`"
                )
                for termo in dict.fromkeys(query_tokens):
                    postings = indice.get(termo, {})
                    df_t = len(postings)
                    idf_t = idfs.get(termo, calcular_idf(n_docs, 0, esquema_idf))
                    st.markdown(f"**Termo `{termo}`** — DF = {df_t}, IDF = {idf_t:.5f}")
                    detalhe = []
                    for doc_id in doc_ids:
                        freq = postings.get(doc_id, 0)
                        tf = calcular_tf(freq, len(docs_tokens[doc_id]), esquema_tf)
                        detalhe.append({
                            "Documento": doc_id,
                            "f(t,d)": freq,
                            "|d|": len(docs_tokens[doc_id]),
                            "TF": round(tf, 5),
                            "IDF": round(idf_t, 5),
                            "TF-IDF": round(tf * idf_t, 5),
                        })
                    st.dataframe(pd.DataFrame(detalhe), hide_index=True,
                                 **LARGURA_TABELA)

            with st.expander("📐 Vetor da consulta (usado na similaridade de cosseno)"):
                st.dataframe(
                    pd.DataFrame(
                        [{"Termo": t, "Peso na query": round(v, 5)}
                         for t, v in vetor_query.items()]
                    ),
                    hide_index=True, **LARGURA_TABELA,
                )
                st.latex(
                    r"\cos(\vec{q},\vec{d}) = "
                    r"\frac{\sum_{t} w_{t,q} \cdot w_{t,d}}"
                    r"{\sqrt{\sum_{t} w_{t,q}^{2}} \cdot \sqrt{\sum_{t} w_{t,d}^{2}}}"
                )

    # =================================================================
    # ABA 4 - FUNDAMENTOS
    # =================================================================
    with aba4:
        st.subheader("Fundamentos Teóricos")

        st.markdown("#### 1. Pipeline de pré-processamento")
        st.markdown(
            "- **Normalização:** minúsculas + remoção de acentos via decomposição "
            "Unicode NFD (descartando marcas `Mn`) + remoção de pontuação.\n"
            "- **Tokenização:** segmentação por expressão regular `[a-z0-9]+`.\n"
            "- **Stopwords:** lista própria de ~110 palavras funcionais do "
            "português, normalizada no mesmo espaço dos tokens.\n"
            "- **Stemming:** stemmer sufixal inspirado no RSLP "
            "(Orengo & Huyck, 2001), implementado do zero com 7 blocos de "
            "regras — plural, feminino, advérbio, aumentativo/diminutivo, "
            "sufixos nominais, sufixos verbais e vogal temática."
        )

        st.markdown("#### 2. Índice invertido")
        st.code(
            "{\n"
            '  "irrig": {"Doc 1": 1, "Doc 5": 1},\n'
            '  "soj":   {"Doc 1": 1, "Doc 2": 1, "Doc 4": 1}\n'
            "}",
            language="json",
        )

        st.markdown("#### 3. Fórmulas")
        st.latex(r"tf(t,d) = \frac{f_{t,d}}{|d|}")
        st.latex(r"idf(t) = \log_{10}\left(\frac{N}{df_t}\right)")
        st.latex(r"tfidf(t,d) = tf(t,d) \times idf(t)")
        st.latex(r"score(q,d) = \sum_{t \in q} tfidf(t,d)")

        st.markdown("#### 4. Por que o IDF pode dar zero?")
        st.markdown(
            "No esquema clássico, se `df = N` o termo aparece em todos os "
            "documentos e `log10(N/N) = 0`. O termo é inútil para "
            "discriminar — comporta-se como uma stopword específica do "
            "domínio. Os esquemas suavizados evitam o zero absoluto."
        )

        st.markdown("#### 5. Limitações conhecidas")
        st.markdown(
            "- O stemmer é heurístico: pode gerar *overstemming* "
            "(radicais distintos colapsam) ou *understemming*.\n"
            "- O modelo é *bag-of-words*: ignora ordem, negação e sinonímia "
            "(\"gotejamento\" e \"irrigação localizada\" não se conectam).\n"
            "- Índice em memória, adequado a um protótipo — não escala para "
            "milhões de documentos sem persistência e compressão."
        )

        st.divider()
        st.caption(
            "Implementação sem scikit-learn, NLTK, spaCy, gensim ou "
            "TfidfVectorizer, conforme a restrição técnica do desafio."
        )


if __name__ == "__main__":
    main()
