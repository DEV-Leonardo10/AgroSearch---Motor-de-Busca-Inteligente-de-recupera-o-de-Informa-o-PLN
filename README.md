# AgroSearch — Motor de Busca Inteligente de Recuperação de Informação (PLN)

Protótipo de motor de busca textual para manuais técnicos de agricultura sustentável, controle de pragas e irrigação.
Todo o núcleo de Recuperação da Informação foi **implementado do zero**, usando apenas a biblioteca padrão do Python
(`re`, `math`, `unicodedata`, `collections`), sem scikit-learn, NLTK, spaCy, gensim ou `TfidfVectorizer`.
Pandas e Streamlit são usados somente na camada de apresentação.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `agrosearch.py` | App Streamlit completo: pré-processamento, índice invertido, TF-IDF, cosseno e interface |
| `relatorio_agrosearch.pdf` | Relatório técnico: arquitetura, fórmulas, validação, divisão de tarefas e limitações |
| `doc1_…pdf` a `doc5_…pdf` | Os cinco documentos da base em PDF, para testar o upload pela interface |
| `requirements.txt` | Dependências |

## Como executar

```bash
pip install -r requirements.txt
streamlit run agrosearch.py
```

`pypdf` só é necessário para o upload de PDFs; a base padrão em texto funciona sem ele.

## Como funciona

1. **Pré-processamento**: normalização (minúsculas, remoção de acentos via NFD, pontuação), tokenização por regex,
   remoção de stopwords (lista própria) e **stemming** sufixal inspirado no RSLP (Orengo & Huyck, 2001).
   Ex.: *irrigação*, *irrigar* e *irrigado* → `irrig`.
2. **Índice invertido**: `{termo: {doc_id: frequência}}`.
3. **Ranqueamento**: TF-IDF acumulado ou similaridade de cosseno (bônus), com esquemas de TF (normalizado, bruto,
   logarítmico) e IDF (clássico, suavizados) selecionáveis na barra lateral.

A interface tem quatro abas — **Pré-processamento**, **Índice Invertido**, **Busca & Ranqueamento** e **Fundamentos** —
e checkboxes para ligar/desligar stopwords e stemming, recalculando vocabulário, índice e ranking em tempo real.
O corpus pode ser editado na barra lateral ou substituído por upload de PDFs/TXTs.

## Validação — consulta "irrigação da soja"

Consulta processada: `[irrig, soj]`, TF normalizado e IDF clássico (N = 5).

| Pos. | Doc | TF-IDF acumulado | Cosseno | Termos casados |
|---|---|---|---|---|
| 1º | Doc 1 | 0,06887 | 0,23921 | irrig, soj |
| 2º | Doc 5 | 0,05685 | 0,20923 | irrig |
| 3º | Doc 2 | 0,03169 | 0,06636 | soj |
| 4º | Doc 4 | 0,02773 | 0,06446 | soj |
| 5º | Doc 3 | 0,00000 | 0,00000 | — |

## Limitações

- O stemmer é heurístico e pode gerar over/understemming.
- Modelo bag-of-words: ignora ordem, negação e sinonímia.
- Índice em memória, adequado ao protótipo.
