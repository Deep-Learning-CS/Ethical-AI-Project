import pandas as pd
import spacy
from collections import defaultdict
import json

def perform_entity_recognition(df, text_column='content', lang_column='lang'):
    """
    Extract named entities from text and link them when possible.
    
    Parameters:
    -----------
    df: pandas DataFrame
        Dataset containing text to process
    text_column: str
        Column name containing the text
    lang_column: str
        Column name indicating language
        
    Returns:
    --------
    DataFrame with added entity columns
    """
    # Load language models
    try:
        nlp_uk = spacy.load("uk_core_news_sm")
        print("Loaded Ukrainian model")
    except:
        print("Ukrainian model not available, downloading...")
        spacy.cli.download("uk_core_news_sm")
        nlp_uk = spacy.load("uk_core_news_sm")
    
    try:
        nlp_ru = spacy.load("ru_core_news_sm")
        print("Loaded Russian model")
    except:
        print("Russian model not available, downloading...")
        spacy.cli.download("ru_core_news_sm")
        nlp_ru = spacy.load("ru_core_news_sm")
    
    # Function to process text based on language
    def extract_entities(text, lang):
        if not isinstance(text, str) or text.strip() == '':
            return []
        
        # Select model based on language
        if lang == 'uk':
            doc = nlp_uk(text)
        elif lang == 'ru':
            doc = nlp_ru(text)
        else:
            return []
        
        # Extract entities with positions
        entities = []
        for ent in doc.ents:
            entity = {
                'text': ent.text,
                'label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            }
            entities.append(entity)
        
        return entities
    
    # Apply entity extraction
    df['entities'] = df.apply(lambda row: extract_entities(row[text_column], row[lang_column]), axis=1)
    
    # Extract entity types and counts
    df['entity_types'] = df['entities'].apply(lambda ents: [e['label'] for e in ents])
    df['entity_count'] = df['entities'].apply(len)
    
    # Create a simple entity linking function (for demonstration)
    # In a real scenario, you would connect to a knowledge base like Wikidata
    def basic_entity_linking(entities):
        linked_entities = []
        
        # Very simple mock KB (in practice, use a real KB API)
        mock_kb = {
            'Україна': {'id': 'Q212', 'description': 'country in Eastern Europe'},
            'Київ': {'id': 'Q1899', 'description': 'capital of Ukraine'},
            'Росія': {'id': 'Q159', 'description': 'country in Eastern Europe and Northern Asia'},
            'Москва': {'id': 'Q649', 'description': 'capital and largest city of Russia'},
            'путін': {'id': 'Q7747', 'description': 'president of Russia'},
            'Зеленський': {'id': 'Q22213928', 'description': 'president of Ukraine'}
        }
        
        for entity in entities:
            entity_text = entity['text']
            if entity_text in mock_kb:
                entity['kb_id'] = mock_kb[entity_text]['id']
                entity['kb_description'] = mock_kb[entity_text]['description']
                linked_entities.append(entity)
            else:
                linked_entities.append(entity)
        
        return linked_entities
    
    # Apply entity linking
    df['linked_entities'] = df['entities'].apply(basic_entity_linking)
    
    # For easier use in analysis, create a JSON string representation
    df['entities_json'] = df['linked_entities'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    
    # Count entity types
    def count_entity_types(entities):
        counts = defaultdict(int)
        for entity in entities:
            counts[entity['label']] += 1
        return dict(counts)
    
    df['entity_type_counts'] = df['entities'].apply(count_entity_types)
    
    return df

# Example usage
if __name__ == "__main__":
    # Assuming df contains your preprocessed data
    from data_read import return_textcl
    df = return_textcl()
    
    # Perform entity recognition and linking
    enriched_df = perform_entity_recognition(df)
    
    # Show sample results
    print("\nSample of entity recognition results:")
    sample = enriched_df[['id', 'entity_count', 'entity_types', 'entities_json']].head(3)
    
    # Pretty print for demonstration
    for idx, row in sample.iterrows():
        print(f"\nID: {row['id']}")
        print(f"Entity count: {row['entity_count']}")
        print(f"Entity types: {row['entity_types']}")
        
        # Parse and print the entities in a readable format
        if row['entity_count'] > 0:
            entities = json.loads(row['entities_json'])
            print("Entities:")
            for e in entities:
                link_info = f" → KB: {e.get('kb_id', 'Not linked')}" if 'kb_id' in e else ""
                print(f"  - {e['text']} ({e['label']}){link_info}")