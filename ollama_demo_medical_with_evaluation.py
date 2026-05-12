# ============================================================
# IMPORTS
# ============================================================

import instructor  # Structured outputs for LLMs
import os
import re
import spacy
import numpy as np

from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._llm import (
    OpenAIEmbeddingService,
    OpenAILLMService
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# LOAD BIOMEDICAL NLP MODEL
# ============================================================

"""
WHY THIS IS IMPORTANT:

Previously:
------------
You manually created:
true_labels = [...]
predicted_labels = [...]

That was synthetic evaluation.

NOW:
-----
We will extract REAL biomedical entities from:
1. Original medical reports
2. GraphRAG generated responses

using biomedical NLP models.

This makes the evaluation REAL and dataset-driven.
"""

# Install before running:
#
# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/en_core_sci_sm-0.5.4.tar.gz

nlp = spacy.load("en_core_sci_sm")

# ============================================================
# DEFINE DOMAIN FOR MEDICAL ANALYSIS
# ============================================================

DOMAIN = """
Analyze these clinical records and identify key medical entities.
Focus on patient demographics, diagnoses, procedures,
lab results, medications, and outcomes.
"""

# ============================================================
# EXAMPLE QUERIES FOR GRAPHRAG
# ============================================================

EXAMPLE_QUERIES = [

    "What are the most common treatments for cardiogenic shock in patients with a history of stroke?",

    "What are the major diagnoses present in the patient records?",

    "Which medications are commonly prescribed for congestive heart failure patients?",

    "Describe complications associated with acute renal failure.",
]

# ============================================================
# DEFINE ENTITY TYPES
# ============================================================

ENTITY_TYPES = [

    "Patient",
    "Diagnosis",
    "Procedure",
    "Lab Test",
    "Medication",
    "Outcome"
]

# ============================================================
# WORKING DIRECTORY
# ============================================================

working_dir = "./WORKING_DIR/mimic_ex500/"

# ============================================================
# INITIALIZE GRAPHRAG
# ============================================================

"""
GRAPH RAG PIPELINE:

Medical Records
      ↓
Embedding Generation
      ↓
Entity Extraction
      ↓
Knowledge Graph Creation
      ↓
Graph Retrieval
      ↓
LLM Reasoning
      ↓
Semantic Answer Generation
"""

grag = GraphRAG(

    working_dir=working_dir,

    n_checkpoints=2,

    domain=DOMAIN,

    example_queries="\n".join(EXAMPLE_QUERIES),

    entity_types=ENTITY_TYPES,

    config=GraphRAG.Config(

        llm_service=OpenAILLMService(

            model="llama3.2",

            base_url="http://localhost:11434/v1",

            api_key="ollama",

            mode=instructor.Mode.JSON,

            client="openai",
        ),

        embedding_service=OpenAIEmbeddingService(

            model="nomic-embed-text",

            base_url="http://localhost:11434/v1",

            api_key="ollama",

            embedding_dim=768,

            client="openai"
        ),
    ),
)

# ============================================================
# EMBEDDING SERVICE
# ============================================================

embedding_service = OpenAIEmbeddingService(

    model="nomic-embed-text",

    base_url="http://localhost:11434/v1",

    api_key="ollama",

    embedding_dim=768,

    client="openai"
)

# ============================================================
# VERIFY EMBEDDINGS
# ============================================================

sample_embedding = embedding_service.get_embedding("test text")

print(f"Embedding dimension: {len(sample_embedding)}")

# ============================================================
# DATASET DIRECTORY
# ============================================================

directory_path = r"E:\semester 4\FastGraphRAG-Medical-Document-Analysis\mimic_ex_500"

# ============================================================
# CLEAN AND STRUCTURE MEDICAL TEXT
# ============================================================

def clean_medical_text(text):

    """
    PURPOSE:
    --------
    Extract important clinical sections from raw reports.

    WHY?
    -----
    Cleaner structured data improves:
    - Graph quality
    - Retrieval quality
    - Entity consistency
    - LLM reasoning
    """

    # Remove excessive spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # Extract important sections
    sections = {

        "medical_history":

            re.search(
                r"past medical history:(.*?)(?:social history:|physical exam:)",
                text,
                re.I
            ),

        "social_history":

            re.search(
                r"social history:(.*?)(?:family history:|physical exam:)",
                text,
                re.I
            ),

        "medications":

            re.search(
                r"medications on admission:(.*?)(?:discharge medications:|discharge disposition:)",
                text,
                re.I
            ),

        "lab_results":

            re.search(
                r"pertinent results:(.*?)(?:brief hospital course:|discharge diagnosis:)",
                text,
                re.I
            ),

        "diagnoses":

            re.search(
                r"discharge diagnosis:(.*?)(?:discharge condition:|discharge instructions:)",
                text,
                re.I
            ),
    }

    structured_data = {

        key: (match.group(1).strip() if match else "")

        for key, match in sections.items()
    }

    return structured_data

# ============================================================
# REAL GROUND TRUTH ENTITY EXTRACTION
# ============================================================

"""
THIS IS THE BIG FIX.

Previously:
-------------
Ground truth labels were manually typed.

NOW:
-----
Ground truth entities are extracted directly
from REAL medical reports using biomedical NLP.
"""

def extract_true_entities(text):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        cleaned_entity = ent.text.strip().lower()

        # Avoid tiny/noisy entities
        if len(cleaned_entity) > 2:

            entities.append(cleaned_entity)

    return list(set(entities))

# ============================================================
# EXTRACT ENTITIES FROM GRAPHRAG RESPONSE
# ============================================================

"""
These are the PREDICTED entities generated by:
- GraphRAG
- Knowledge Graph Retrieval
- LLM reasoning
"""

def extract_predicted_entities(response):

    doc = nlp(response)

    predicted_entities = []

    for ent in doc.ents:

        cleaned_entity = ent.text.strip().lower()

        if len(cleaned_entity) > 2:

            predicted_entities.append(cleaned_entity)

    return list(set(predicted_entities))

# ============================================================
# INSERT DOCUMENTS INTO GRAPHRAG
# ============================================================

def graph_index(directory_path):

    file_count = 0

    total_files = sum(
        1 for f in os.listdir(directory_path)
        if f.endswith(".txt")
    )

    for filename in os.listdir(directory_path):

        if filename.endswith('.txt'):

            file_path = os.path.join(directory_path, filename)

            try:

                with open(file_path, 'r', encoding='utf-8') as file:

                    content = file.read()

                    # Clean report
                    structured_data = clean_medical_text(content)

                    # Replace missing values
                    for key in structured_data:

                        structured_data[key] = (
                            structured_data[key]
                            or
                            "Unknown"
                        )

                    # Insert into GraphRAG
                    grag.insert(str(structured_data))

                file_count += 1

                print("*******************************************")
                print(f"Processed Files: {file_count}/{total_files}")
                print("*******************************************")

            except Exception as e:

                print(f"Error processing {filename}: {e}")

# ============================================================
# INDEX DATASET
# ============================================================

graph_index(directory_path)

# ============================================================
# SAVE GRAPHML FOR NEO4J
# ============================================================

print("**********************************************")

os.makedirs("neo4j_graph", exist_ok=True)

grag.save_graphml(

    output_path=
    "neo4j_graph/oxford_graph_chunk_entity_relation.graphml"
)

# ============================================================
# SAMPLE QUERIES
# ============================================================

queries = [

    "What are the common risk factors for sepsis in ICU patients?",

    "How do trends in lab results correlate with patient outcomes in cases of acute kidney injury?",

    "Describe the sequence of interventions for patients undergoing major cardiac surgery.",

    "How do patient demographics and comorbidities influence treatment decisions in the ICU?",

    "What patterns of medication usage are observed among patients with chronic obstructive pulmonary disease (COPD)?"
]

# ============================================================
# QUERY GRAPHRAG
# ============================================================

print("\n================ GRAPH RAG RESPONSES ================\n")

all_true_labels = []
all_predicted_labels = []

hallucinated_relations = 0
total_generated_relations = 0

# ============================================================
# REAL EVALUATION LOOP
# ============================================================

"""
REAL EVALUATION PIPELINE:

Original Medical Record
            vs
GraphRAG Generated Response

This is now REAL evaluation.
"""

for filename in os.listdir(directory_path):

    if filename.endswith(".txt"):

        file_path = os.path.join(directory_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:

            original_text = f.read()

        # ====================================================
        # EXTRACT REAL GROUND TRUTH ENTITIES
        # ====================================================

        true_entities = extract_true_entities(original_text)

        # ====================================================
        # RUN ALL SAMPLE QUERIES
        # ====================================================

        """
        IMPORTANT CHANGE:
        -----------------
        Previously:
        A random/custom query was created:
        
        "What are the diagnoses, medications..."

        NOW:
        -----
        We use ONLY the official queries
        already defined in your project.

        This keeps:
        - evaluation aligned with project goals
        - query consistency
        - cleaner benchmarking
        - realistic GraphRAG testing
        """

        for query in queries:

            # =================================================
            # ASK GRAPHRAG QUESTION
            # =================================================

            result = grag.query(query)

            print(f"\nQuestion: {query}")

            print("\nGenerated Response:")

            print(result.response)

            # ================================================
            # EXTRACT PREDICTED ENTITIES
            # ================================================

            predicted_entities = extract_predicted_entities(
                result.response
            )

            # ================================================
            # CONVERT TO BINARY LABELS
            # ================================================

            all_entities = list(
                set(true_entities + predicted_entities)
            )

            y_true = [

                1 if entity in true_entities else 0

                for entity in all_entities
            ]

            y_pred = [

                1 if entity in predicted_entities else 0

                for entity in all_entities
            ]

            # Store globally for final metrics
            all_true_labels.extend(y_true)

            all_predicted_labels.extend(y_pred)

            # ================================================
            # REAL HALLUCINATION DETECTION
            # ================================================

            """
            LOGIC:
            -------
            If generated entity is absent
            in original medical report,
            treat as hallucination.
            """

            for entity in predicted_entities:

                total_generated_relations += 1

                if entity not in original_text.lower():

                    hallucinated_relations += 1

# ============================================================
# CALCULATE REAL EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    all_true_labels,
    all_predicted_labels
)

precision = precision_score(
    all_true_labels,
    all_predicted_labels,
    zero_division=0
)

recall = recall_score(
    all_true_labels,
    all_predicted_labels,
    zero_division=0
)

f1 = f1_score(
    all_true_labels,
    all_predicted_labels,
    zero_division=0
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

conf_matrix = confusion_matrix(
    all_true_labels,
    all_predicted_labels
)

# Safe extraction
if conf_matrix.shape == (2, 2):

    tn, fp, fn, tp = conf_matrix.ravel()

else:

    tn = fp = fn = tp = 0

# ============================================================
# FALSE POSITIVE RATE
# ============================================================

false_positive_rate = (

    fp / (fp + tn)

    if (fp + tn) > 0

    else 0
)

# ============================================================
# HALLUCINATION RATE
# ============================================================

hallucination_rate = (

    hallucinated_relations / total_generated_relations

    if total_generated_relations > 0

    else 0
) * 100

# ============================================================
# FINAL RESULTS
# ============================================================

print("================ NLP METRICS ====================")

print(f"Accuracy                 : {accuracy * 100:.2f}%")

print(f"Precision                : {precision * 100:.2f}%")

print(f"Recall                   : {recall * 100:.2f}%")

print(f"F1 Score                 : {f1 * 100:.2f}%")

print(f"False Positive Rate      : {false_positive_rate:.2f}")

print(f"Hallucination Rate       : {hallucination_rate:.2f}%")

print("====================================================")

# ============================================================
# WHY THIS EVALUATION IS NOW REAL
# ============================================================

"""
BEFORE:
--------
- Manual labels
- Synthetic examples
- Artificial relations

NOW:
-----
- Ground truth extracted from REAL reports
- Predictions extracted from REAL GraphRAG responses
- Metrics computed dynamically
- Hallucinations verified against source text

This is now REAL evaluation.
"""

# ============================================================
# HALLUCINATION FORMULA
# ============================================================

"""
hallucination_rate =

(
    hallucinated_relations
    /
    total_generated_relations
) * 100
"""
