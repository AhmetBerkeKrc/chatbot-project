from sentence_transformers import SentenceTransformer
import numpy as np
from datasets import load_dataset
import faiss

# Load the dataset from Hugging Face
dataset = load_dataset("codexist/medical_data")

# Load the embedding model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Extract text data
texts = [row["data"] for row in dataset["train"]]

# Convert text data to embeddings
vectors = model.encode(texts, show_progress_bar=True)

# Initialize FAISS index for 384-dimensional vectors
faiss_index = faiss.IndexFlatL2(384)

# Convert embeddings to float32 and add to the FAISS index
faiss_index.add(np.array(vectors).astype(np.float32))

# Save the FAISS index for later use
faiss.write_index(faiss_index, "medical_faiss.index")

def search_faiss(query: str, top_k: int = 5):
    # Convert query to vector using the model
    query_vector = model.encode([query]).astype(np.float32)

    # Perform search in FAISS index
    distances, indices = faiss_index.search(query_vector, top_k)

    # Display results
    print("\nTop results for:", query)
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {texts[idx]} (Score: {distances[0][i]})")

# Example query
search_faiss("What is hydrops fetalis?")


# RESULT: 
# Top results for: What is hydrops fetalis?
# 1. Hydrops fetalis is a serious condition in which abnormal amounts of fluid build up in two or more body areas of a fetus or newborn. There are two types of hydrops fetalis: immune and nonimmune. Immune hydrops fetalis 
# is a complication of a severe form of Rh incompatibility. Rh compatibility causes massive red blood cell destruction, which leads to several problems, including total body swelling. Severe swelling can interfere with how the body organs work. Nonimmune hydrops fetalis occurs when a disease or medical condition disrupts the body's ability to manage fluid. There are three main causes for this type: heart or lung problems, severe anemia (thalassemia), and genetic defects, including Turner syndrome. The exact cause depends on which form a baby has. (Score: 24.651775360107422)
# 2. Fetal cystic hygroma is a congenital malformation of the lymphatic system. The lymphatic system is a network of vessels that maintains fluids in the blood, as well as transports fats and immune system cells. Cystic hygromas are single or multiple cysts found mostly in the neck region. In the fetus, a cystic hygroma can progress to hydrops (an excess amount of fluid in the body) and eventually lead to fetal death. Some cases resolve leading to webbed neck, edema (swelling), and a lymphangioma (a benign yellowish-tan tumor on the skin composed of swollen lymph vessels). In other instances, the hygroma can progress in size to become larger than the fetus. Cystic hygromas can be classified as septated (multiloculated) or nonseptated (simple). Cystic hygromas can occur as an isolated finding or in association with other birth defects as part of a syndrome (chromosomal abnormalities or syndromes caused by gene mutations). They may result from environmental factors (maternal virus infection or alcohol abuse during pregnancy), genetic factors, or unknown factors. The majority of prenatally diagnosed cystic hygromas are associated with Turner syndrome or other chromosomal abnormalities like trisomy 21. Isolated cystic hygroma can be inherited as an autosomal recessive disorder. Fetal cystic hygroma have being treated with OK-432, a lyophilized mixture of Group A Streptococcus pyogenes and benzyl penicillin, and with serial thoracocentesis plus paracentesis. (Score: 38.87041473388672)
# 3. When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis.

# Hydronephrosis is the most common problem found during prenatal ultrasound of a baby in the womb. The swelling may be easy to see or barely detectable. The results of hydronephrosis may be mild or severe, yet the long-term outcome for the childs health cannot always be predicted by the severity of swelling. Urine blockage may damage the developing kidneys and reduce their ability to filter. In the most severe cases of urine blockage, where little or no urine leaves the babys bladder, the amount of amniotic fluid is reduced to the point that the babys lung development is threatened.

# After birth, urine blockage may raise a childs risk of developing a UTI. Recurring UTIs can lead to more permanent kidney damage. (Score: 39.7662353515625)
# 4. HEM (hydrops fetalis, ectopic calcifications, "moth-eaten" skeletal dysplasia) is a very rare type of lethal skeletal dysplasia. According to the reported cases of HEM in the medical literature, the condition's main features are hydrops fetalis, dwarfism with severely shortened limbs and relatively normal-sized hands and feet, a "moth-eaten" appearance of the skeleton, flat vertebral bodies and ectopic calcifications. HEM is an autosomal recessive condition caused by a mutation in the lamin B receptor (LBR) gene. No treatment or cure is currently known for HEM. (Score: 40.24720764160156)
# 5. Hydromyelia refers to an abnormal widening of the central canal of the spinal cord that creates a cavity in which cerebrospinal fluid (commonly known as spinal fluid) can accumulate. As spinal fluid builds up, it may 
# put abnormal pressure on the spinal cord and damage nerve cells and their connections. Hydromyelia is sometimes used interchangeably with syringomyelia, the name for a condition that also involves cavitation in the spinal cord. In hydromyelia, the cavity that forms is connected to the fourth ventricle in the brain, and is almost always associated in infants and children with hydrocephalus or birth defects such as Chiari Malformation II 
# and Dandy-Walker syndrome. Syringomyelia, however, features a closed cavity and occurs primarily in adults, the majority of whom have Chiari Malformation type 1 or have experienced spinal cord trauma. Symptoms, which may occur over time, include weakness of the hands and arms, stiffness in the legs; and sensory loss in the neck and arms. Some individuals have severe pain in the neck and arms. Diagnosis is made by magnetic resonance imaging (MRI), which reveals abnormalities in the anatomy of the spinal cord.. (Score: 40.698577880859375)
