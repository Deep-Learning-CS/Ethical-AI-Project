import pandas as pd
import spacy
from spacy.tokens import Doc
import json
from nltk.tree import Tree
import nltk

class SimpleSemanticRoleLabeler:
    """
    A simplified semantic role labeler using dependency parsing
    to extract basic predicate-argument structures.
    """
    
    def __init__(self, lang):
        self.lang = lang
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Load appropriate spaCy model
        if lang == 'uk':
            try:
                self.nlp = spacy.load("uk_core_news_sm")
            except:
                spacy.cli.download("uk_core_news_sm")
                self.nlp = spacy.load("uk_core_news_sm")
        elif lang == 'ru':
            try:
                self.nlp = spacy.load("ru_core_news_sm")
            except:
                spacy.cli.download("ru_core_news_sm")
                self.nlp = spacy.load("ru_core_news_sm")
        else:
            raise ValueError(f"Unsupported language: {lang}")
    
    def extract_predicates(self, doc):
        """Extract predicates (verbs) and their arguments."""
        predicates = []
        
        for token in doc:
            # Find verb tokens that are the root or have a subject
            if token.pos_ == "VERB" and (token.dep_ == "ROOT" or any(child.dep_ in ["nsubj", "csubj"] for child in token.children)):
                predicate = {
                    "predicate": token.text,
                    "predicate_lemma": token.lemma_,
                    "position": (token.idx, token.idx + len(token.text)),
                    "arguments": self._extract_arguments(token)
                }
                predicates.append(predicate)
                
        return predicates
    
    def _extract_arguments(self, verb_token):
        """Extract arguments for a given predicate."""
        arguments = []
        
        # A mapping of dependency relations to semantic roles
        role_mapping = {
            "nsubj": "AGENT",      # subject → agent
            "dobj": "PATIENT",     # direct object → patient
            "iobj": "RECIPIENT",   # indirect object → recipient
            "pobj": "LOCATION",    # object of preposition → can be various roles
            "obl": "INSTRUMENT",   # oblique → often instrument or manner
            "advmod": "MANNER",    # adverbial modifier → manner
            "tmod": "TIME",        # temporal modifier → time
        }
        
        for child in verb_token.children:
            # Determine the role based on dependency relation
            role = role_mapping.get(child.dep_, child.dep_)
            
            # For prepositional phrases, include the entire phrase
            if child.dep_ == "prep":
                prep_objs = list(child.children)
                if prep_objs:
                    for pobj in prep_objs:
                        if pobj.dep_ == "pobj":
                            # Get the full prepositional phrase
                            prep_phrase = self._get_subtree_span(pobj)
                            arguments.append({
                                "text": child.text + " " + prep_phrase,
                                "role": self._determine_pp_role(child.text),
                                "position": (child.idx, child.idx + len(child.text) + 1 + len(prep_phrase))
                            })
            else:
                # Get full span of the argument (including its children)
                arg_text = self._get_subtree_span(child)
                arguments.append({
                    "text": arg_text,
                    "role": role,
                    "position": (child.idx, child.idx + len(arg_text))
                })
                
        return arguments
    
    def _get_subtree_span(self, token):
        """Get the text span of a token and all its children."""
        if not list(token.children):
            return token.text
            
        # Sort all descendant tokens by their position in the sentence
        span_tokens = sorted([token] + list(token.subtree), key=lambda t: t.i)
        span_text = ' '.join(t.text for t in span_tokens)
        return span_text
    
    def _determine_pp_role(self, preposition):
        """Determine semantic role based on preposition."""
        # Simplified role mapping for common prepositions
        # This would need to be expanded for better coverage
        if self.lang == 'uk':
            if preposition in ["в", "у", "на"]:
                return "LOCATION"
            elif preposition in ["з", "із"]:
                return "SOURCE"
            elif preposition in ["до", "для"]:
                return "DESTINATION"
            elif preposition in ["через"]:
                return "CAUSE"
        elif self.lang == 'ru':
            if preposition in ["в", "на"]:
                return "LOCATION"
            elif preposition in ["из", "с"]:
                return "SOURCE"
            elif preposition in ["к", "для"]:
                return "DESTINATION"
            elif preposition in ["из-за"]:
                return "CAUSE"
                
        return "MODIFIER"  # Default role
        
    def process_text(self, text):
        """Process text and extract semantic roles."""
        if not isinstance(text, str) or text.strip() == '':
            return []
            
        doc = self.nlp(text)
        return self.extract_predicates(doc)
        
    def visualize_dependencies(self, text):
        """Generate a visual representation of dependencies."""
        doc = self.nlp(text)
        return self._to_nltk_tree(doc[0].root)
    
    def _to_nltk_tree(self, root):
        """Convert spaCy dependency tree to NLTK tree for visualization."""
        def get_subtree(token):
            return Tree(f"{token.text}/{token.dep_}", [get_subtree(child) for child in token.children])
        return get_subtree(root)

def apply_semantic_role_labeling(df, text_column='content', lang_column='lang'):
    """
    Apply semantic role labeling to text in a dataframe.
    
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
    DataFrame with added semantic role columns
    """
    # Group by language to avoid reloading models
    results = []
    
    for lang in df[lang_column].unique():
        print(f"Processing {lang} texts...")
        lang_df = df[df[lang_column] == lang].copy()
        
        # Initialize SRL for this language
        srl = SimpleSemanticRoleLabeler(lang)
        
        # Process each text
        lang_df['semantic_predicates'] = lang_df[text_column].apply(srl.process_text)
        lang_df['semantic_predicate_count'] = lang_df['semantic_predicates'].apply(len)
        
        # For easier analysis and storage
        lang_df['semantic_predicates_json'] = lang_df['semantic_predicates'].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        
        results.append(lang_df)
    
    # Combine results
    result_df = pd.concat(results, ignore_index=True)
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Assuming df contains your preprocessed data
    from data_read import return_textcl
    df = return_textcl()
    
    # Only process a small sample for demonstration
    sample_df = df.head(5)
    
    # Apply semantic role labeling
    labeled_df = apply_semantic_role_labeling(sample_df)
    
    # Show sample results
    print("\nSample of semantic role labeling results:")
    sample = labeled_df[['id', 'semantic_predicate_count', 'semantic_predicates_json']].head(2)
    
    # Pretty print for demonstration
    for idx, row in sample.iterrows():
        print(f"\nID: {row['id']}")
        print(f"Predicate count: {row['semantic_predicate_count']}")
        
        if row['semantic_predicate_count'] > 0:
            predicates = json.loads(row['semantic_predicates_json'])
            print("Predicates and arguments:")
            for i, pred in enumerate(predicates):
                print(f"  {i+1}. {pred['predicate']} (lemma: {pred['predicate_lemma']})")
                for arg in pred['arguments']:
                    print(f"     - {arg['role']}: {arg['text']}")