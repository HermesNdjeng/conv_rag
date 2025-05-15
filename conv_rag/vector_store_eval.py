#!/usr/bin/env python3
"""
Script d'évaluation des systèmes de stockage vectoriel pour le projet Conv-RAG.
Compare les performances de FAISS et Chroma sur plusieurs métriques.
"""
import os
import sys
import logging
import argparse
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
import json

from retriever import DocumentRetriever, RetrievalConfig
from generation import RAGGenerator, GeneratorConfig
from utils.logging_utils import setup_logger

# Configuration du logger
logger = setup_logger("eval")

def evaluate_vector_stores(test_queries, embedding_model=None):
    """
    Compare FAISS et Chroma sur le même ensemble de requêtes.
    
    Args:
        test_queries: Liste de requêtes pour tester les systèmes
        embedding_model: Modèle d'embedding à utiliser (facultatif)
        
    Returns:
        DataFrame contenant les métriques comparatives
    """
    logger.info(f"Évaluation des performances de récupération sur {len(test_queries)} requêtes...")
    results = []
    
    # Créer deux retrievers identiques mais avec des vector stores différents
    faiss_retriever = DocumentRetriever(
        config=RetrievalConfig(index_type="faiss", top_k=8, score_threshold=None)
    )
    chroma_retriever = DocumentRetriever(
        config=RetrievalConfig(index_type="chroma", top_k=8, score_threshold=None)
    )
    
    # Utiliser l'embedding model du projet
    embedding_model = faiss_retriever.indexer.embeddings
    logger.info(f"Utilisation du modèle d'embedding: {type(embedding_model).__name__}")
    
    for i, query in enumerate(tqdm(test_queries, desc="Évaluation des vector stores")):
        logger.debug(f"Traitement de la requête {i+1}/{len(test_queries)}: '{query}'")
        
        # Récupérer les documents de chaque vector store
        start_time_faiss = time.time()
        faiss_result = faiss_retriever.retrieve(query)
        faiss_time = time.time() - start_time_faiss
        
        start_time_chroma = time.time()
        chroma_result = chroma_retriever.retrieve(query)
        chroma_time = time.time() - start_time_chroma
        
        # Calculer l'embedding de la requête
        query_embedding = embedding_model.embed_query(query)
        
        # Calculer le score moyen de similarité pour FAISS
        faiss_docs = faiss_result.documents
        faiss_similarities = faiss_result.scores
        faiss_avg_similarity = np.mean(faiss_similarities) if faiss_similarities else 0
        
        # Calculer le score moyen de similarité pour Chroma
        chroma_docs = chroma_result.documents
        chroma_similarities = chroma_result.scores
        chroma_avg_similarity = np.mean(chroma_similarities) if chroma_similarities else 0
        
        # Comparer le contenu des chunks récupérés
        overlap = len(set([doc.page_content for doc in faiss_docs]) & 
                     set([doc.page_content for doc in chroma_docs]))
        overlap_percent = overlap / 8 * 100 if faiss_docs and chroma_docs else 0
        
        logger.debug(f"Requête: '{query}' | Similarité FAISS: {faiss_avg_similarity:.4f} | "
                    f"Similarité Chroma: {chroma_avg_similarity:.4f} | "
                    f"Chevauchement: {overlap_percent:.1f}%")
        
        # Stocker les résultats
        results.append({
            'query': query,
            'faiss_time': faiss_time,
            'chroma_time': chroma_time,
            'faiss_avg_similarity': faiss_avg_similarity,
            'chroma_avg_similarity': chroma_avg_similarity,
            'overlap_percent': overlap_percent,
            'faiss_doc_count': len(faiss_docs),
            'chroma_doc_count': len(chroma_docs),
        })
    
    result_df = pd.DataFrame(results)
    logger.info("Évaluation des vector stores terminée.")
    logger.info(f"Temps moyen FAISS: {result_df['faiss_time'].mean():.4f}s | "
               f"Temps moyen Chroma: {result_df['chroma_time'].mean():.4f}s")
    logger.info(f"Similarité moyenne FAISS: {result_df['faiss_avg_similarity'].mean():.4f} | "
               f"Similarité moyenne Chroma: {result_df['chroma_avg_similarity'].mean():.4f}")
    return result_df

def evaluate_answer_quality(test_questions, generator_config):
    """
    Compare la qualité des réponses générées entre FAISS et Chroma.
    
    Args:
        test_questions: Liste de questions pour tester les systèmes
        generator_config: Configuration du générateur RAG
        
    Returns:
        DataFrame contenant les métriques comparatives
    """
    logger.info(f"Évaluation de la qualité des réponses sur {len(test_questions)} questions...")
    results = []
    
    # Générer des réponses avec les deux vector stores
    for i, question in enumerate(tqdm(test_questions, desc="Évaluation des réponses")):
        logger.debug(f"Traitement de la question {i+1}/{len(test_questions)}: '{question}'")
        
        # Configuration avec FAISS
        faiss_retriever = DocumentRetriever(
            config=RetrievalConfig(index_type="faiss", top_k=8)
        )
        faiss_generator = RAGGenerator(
            config=generator_config,
            retriever=faiss_retriever
        )
        
        # Configuration avec Chroma
        chroma_retriever = DocumentRetriever(
            config=RetrievalConfig(index_type="chroma", top_k=8)
        )
        chroma_generator = RAGGenerator(
            config=generator_config,
            retriever=chroma_retriever
        )
        
        # Générer les réponses
        logger.debug(f"Génération de la réponse avec FAISS pour: '{question}'")
        faiss_result = faiss_generator.generate(question)
        
        logger.debug(f"Génération de la réponse avec Chroma pour: '{question}'")
        chroma_result = chroma_generator.generate(question)
        
        faiss_token_count = faiss_result.token_usage.get('total_tokens', 0) if faiss_result.token_usage else 0
        chroma_token_count = chroma_result.token_usage.get('total_tokens', 0) if chroma_result.token_usage else 0
        
        # Collecter les métriques
        results.append({
            'question': question,
            'faiss_answer': faiss_result.answer,
            'chroma_answer': chroma_result.answer,
            'faiss_token_count': faiss_token_count,
            'chroma_token_count': chroma_token_count,
            'answer_match': faiss_result.answer == chroma_result.answer,
        })
        
        logger.debug(f"Tokens FAISS: {faiss_token_count} | Tokens Chroma: {chroma_token_count}")
    
    result_df = pd.DataFrame(results)
    logger.info("Évaluation des réponses terminée.")
    logger.info(f"Consommation moyenne de tokens - FAISS: {result_df['faiss_token_count'].mean():.1f} | "
               f"Chroma: {result_df['chroma_token_count'].mean():.1f}")
    logger.info(f"Pourcentage de réponses identiques: {(result_df['answer_match'].mean() * 100):.1f}%")
    
    return result_df

def evaluate_chunk_relevance(test_queries, ground_truth=None):
    """
    Évalue la pertinence des chunks récupérés pour chaque requête.
    
    Args:
        test_queries: Liste de requêtes pour tester les systèmes
        ground_truth: Vérité terrain (facultatif)
        
    Returns:
        DataFrame contenant les métriques de pertinence
    """
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    
    logger.info(f"Évaluation de la pertinence des chunks sur {len(test_queries)} requêtes...")
    
    # Créer les retrievers
    faiss_retriever = DocumentRetriever(config=RetrievalConfig(index_type="faiss", top_k=5))
    chroma_retriever = DocumentRetriever(config=RetrievalConfig(index_type="chroma", top_k=5))
    
    # Créer un évaluateur LLM
    logger.info("Initialisation du modèle évaluateur GPT-4...")
    evaluator = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    eval_prompt = PromptTemplate(
        template="""
        [INSTRUCTION]
        Évalue la pertinence du texte pour répondre à la question. 
        Note de 1 à 10 où 10 est parfaitement pertinent et 1 est complètement hors-sujet.
        [/INSTRUCTION]
        
        Question: {query}
        
        Texte à évaluer:
        {chunk}
        
        Ton évaluation (uniquement le score de 1 à 10):
        """,
        input_variables=["query", "chunk"]
    )
    
    results = []
    
    for i, query in enumerate(tqdm(test_queries, desc="Évaluation de la pertinence")):
        logger.debug(f"Évaluation de la pertinence pour la requête {i+1}/{len(test_queries)}: '{query}'")
        
        faiss_chunks = faiss_retriever.retrieve(query).documents
        chroma_chunks = chroma_retriever.retrieve(query).documents
        
        # Évaluer la pertinence de chaque chunk
        faiss_scores = []
        logger.debug(f"Évaluation de {len(faiss_chunks)} chunks FAISS...")
        for j, chunk in enumerate(faiss_chunks):
            logger.debug(f"Évaluation du chunk FAISS {j+1}/{len(faiss_chunks)}")
            result = evaluator.predict(eval_prompt.format(query=query, chunk=chunk.page_content))
            try:
                score = int(result.strip())
                faiss_scores.append(score)
                logger.debug(f"Score de pertinence FAISS {j+1}: {score}/10")
            except:
                logger.warning(f"Impossible de convertir le score pour le chunk FAISS {j+1}: {result}")
                faiss_scores.append(0)
        
        chroma_scores = []
        logger.debug(f"Évaluation de {len(chroma_chunks)} chunks Chroma...")
        for j, chunk in enumerate(chroma_chunks):
            logger.debug(f"Évaluation du chunk Chroma {j+1}/{len(chroma_chunks)}")
            result = evaluator.predict(eval_prompt.format(query=query, chunk=chunk.page_content))
            try:
                score = int(result.strip())
                chroma_scores.append(score)
                logger.debug(f"Score de pertinence Chroma {j+1}: {score}/10")
            except:
                logger.warning(f"Impossible de convertir le score pour le chunk Chroma {j+1}: {result}")
                chroma_scores.append(0)
        
        faiss_avg = np.mean(faiss_scores) if faiss_scores else 0
        chroma_avg = np.mean(chroma_scores) if chroma_scores else 0
        
        logger.info(f"Requête: '{query}' | Pertinence moyenne FAISS: {faiss_avg:.2f}/10 | "
                  f"Pertinence moyenne Chroma: {chroma_avg:.2f}/10")
        
        results.append({
            'query': query,
            'faiss_relevance': faiss_avg,
            'chroma_relevance': chroma_avg,
            'faiss_max_relevance': max(faiss_scores) if faiss_scores else 0,
            'chroma_max_relevance': max(chroma_scores) if chroma_scores else 0,
            'faiss_min_relevance': min(faiss_scores) if faiss_scores else 0,
            'chroma_min_relevance': min(chroma_scores) if chroma_scores else 0,
        })
    
    result_df = pd.DataFrame(results)
    logger.info("Évaluation de la pertinence terminée.")
    logger.info(f"Pertinence moyenne - FAISS: {result_df['faiss_relevance'].mean():.2f}/10 | "
               f"Chroma: {result_df['chroma_relevance'].mean():.2f}/10")
    
    return result_df

def plot_comparison(df, metric_cols, title, output_dir="results"):
    """
    Génère un graphique de comparaison pour les métriques spécifiées.
    
    Args:
        df: DataFrame contenant les données
        metric_cols: Liste des colonnes à comparer
        title: Titre du graphique
        output_dir: Répertoire de sortie pour sauvegarder les graphiques
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    df[metric_cols].mean().plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
    plt.title(title, fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(df[metric_cols].mean()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=11)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Graphique sauvegardé: {output_file}")
    plt.close()

def main():
    """
    Fonction principale d'évaluation des vector stores.
    """
    parser = argparse.ArgumentParser(description="Évaluation des performances de FAISS vs Chroma")
    parser.add_argument("--output-dir", default="results", help="Répertoire de sortie pour les résultats")
    parser.add_argument("--similarity-only", action="store_true", help="Exécuter uniquement l'évaluation de similarité")
    parser.add_argument("--quality-only", action="store_true", help="Exécuter uniquement l'évaluation de qualité des réponses")
    parser.add_argument("--relevance-only", action="store_true", help="Exécuter uniquement l'évaluation de pertinence")
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Démarrage de l'évaluation des vector stores (FAISS vs Chroma)")
    logger.info(f"Les résultats seront sauvegardés dans: {output_dir}")
    
    # Créer un ensemble de requêtes variées pour tester différents aspects
    test_questions = [
        # Questions factuelles
        "Qui était Ruben Um Nyobe?",
        "En quelle année a été fondée l'UPC?",
        "Où Ruben Um Nyobe a-t-il vécu?",
        
        # Questions complexes
        "Quelles étaient les relations entre l'UPC et le gouvernement français?",
        "Comment la lutte pour l'indépendance a-t-elle évolué après la mort d'Um Nyobe?",
        
        # Questions nécessitant une analyse
        "Pourquoi l'UPC était-elle considérée comme communiste par certains?",
        "Quelles étaient les principales revendications de l'UPC?",
        
        # Questions hors domaine (pour tester la robustesse)
        "Que dit Um Nyobe sur l'économie camerounaise?",
        "Comment fonctionnait l'organisation interne de l'UPC?",
    ]
    
    # Sauvegarder les questions dans un fichier
    with open(os.path.join(output_dir, "test_questions.json"), "w", encoding="utf-8") as f:
        json.dump(test_questions, f, ensure_ascii=False, indent=2)
    
    # Exécuter les évaluations selon les arguments
    if args.similarity_only or not (args.quality_only or args.relevance_only):
        logger.info("=== ÉVALUATION DE SIMILARITÉ ===")
        similarity_results = evaluate_vector_stores(test_questions)
        similarity_results.to_csv(os.path.join(output_dir, "similarity_results.csv"), index=False)
        
        plot_comparison(similarity_results, ['faiss_avg_similarity', 'chroma_avg_similarity'], 
                       'Similarité Moyenne des Chunks', output_dir)
        plot_comparison(similarity_results, ['faiss_time', 'chroma_time'], 
                       'Temps de Récupération (secondes)', output_dir)
        plot_comparison(similarity_results, ['overlap_percent'], 
                       'Chevauchement des Résultats (%)', output_dir)
    
    if args.quality_only or not (args.similarity_only or args.relevance_only):
        logger.info("=== ÉVALUATION DE QUALITÉ DES RÉPONSES ===")
        generator_config = GeneratorConfig(model_name="gpt-3.5-turbo", temperature=0.0)
        quality_results = evaluate_answer_quality(test_questions[:3], generator_config)  # Limité à 3 pour l'efficacité
        quality_results.to_csv(os.path.join(output_dir, "quality_results.csv"), index=False)
        
        # Sauvegarder les réponses complètes
        with open(os.path.join(output_dir, "comparison_answers.json"), "w", encoding="utf-8") as f:
            responses = []
            for _, row in quality_results.iterrows():
                responses.append({
                    "question": row['question'],
                    "faiss_answer": row['faiss_answer'],
                    "chroma_answer": row['chroma_answer']
                })
            json.dump(responses, f, ensure_ascii=False, indent=2)
        
        plot_comparison(quality_results, ['faiss_token_count', 'chroma_token_count'], 
                       'Consommation de Tokens', output_dir)
    
    if args.relevance_only or not (args.similarity_only or args.quality_only):
        logger.info("=== ÉVALUATION DE PERTINENCE DES CHUNKS ===")
        relevance_results = evaluate_chunk_relevance(test_questions[:3])  # Limité à 3 pour économiser GPT-4
        relevance_results.to_csv(os.path.join(output_dir, "relevance_results.csv"), index=False)
        
        plot_comparison(relevance_results, ['faiss_relevance', 'chroma_relevance'], 
                       'Pertinence Moyenne des Chunks', output_dir)
        plot_comparison(relevance_results, ['faiss_max_relevance', 'chroma_max_relevance'], 
                       'Pertinence Max des Chunks', output_dir)
    
    logger.info(f"Évaluation terminée. Tous les résultats ont été sauvegardés dans: {output_dir}")
    
    # Générer un rapport résumé
    with open(os.path.join(output_dir, "evaluation_summary.txt"), "w") as f:
        f.write("RÉSUMÉ DE L'ÉVALUATION FAISS VS CHROMA\n")
        f.write("=====================================\n\n")
        f.write(f"Date de l'évaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nombre de questions testées: {len(test_questions)}\n\n")
        
        if 'similarity_results' in locals():
            f.write("MÉTRIQUES DE SIMILARITÉ\n")
            f.write(f"Similarité moyenne FAISS: {similarity_results['faiss_avg_similarity'].mean():.4f}\n")
            f.write(f"Similarité moyenne Chroma: {similarity_results['chroma_avg_similarity'].mean():.4f}\n")
            f.write(f"Temps moyen FAISS: {similarity_results['faiss_time'].mean():.4f}s\n")
            f.write(f"Temps moyen Chroma: {similarity_results['chroma_time'].mean():.4f}s\n")
            f.write(f"Chevauchement moyen: {similarity_results['overlap_percent'].mean():.2f}%\n\n")
        
        if 'relevance_results' in locals():
            f.write("MÉTRIQUES DE PERTINENCE\n")
            f.write(f"Pertinence moyenne FAISS: {relevance_results['faiss_relevance'].mean():.2f}/10\n")
            f.write(f"Pertinence moyenne Chroma: {relevance_results['chroma_relevance'].mean():.2f}/10\n")
            f.write(f"Pertinence maximale moyenne FAISS: {relevance_results['faiss_max_relevance'].mean():.2f}/10\n")
            f.write(f"Pertinence maximale moyenne Chroma: {relevance_results['chroma_max_relevance'].mean():.2f}/10\n\n")
        
        if 'quality_results' in locals():
            f.write("MÉTRIQUES DE QUALITÉ DES RÉPONSES\n")
            f.write(f"Tokens moyens FAISS: {quality_results['faiss_token_count'].mean():.1f}\n")
            f.write(f"Tokens moyens Chroma: {quality_results['chroma_token_count'].mean():.1f}\n")
            f.write(f"Pourcentage de réponses identiques: {(quality_results['answer_match'].mean() * 100):.1f}%\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())